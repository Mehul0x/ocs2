#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <ocs2_core/Types.h>
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/MRT_BASE.h"

struct MPCSolutionPointers {
    ocs2::PrimalSolution primalSolution;
    ocs2::CommandData command;
    ocs2::PerformanceIndex performanceIndices;
};

/**
    * An entry in the MPC cache.
 */
struct MPCCacheEntry_ocs {
    ocs2::vector_t feat;
    MPCSolutionPointers ptr;
    double trust_radius = 10.0;
    uint64_t last_use = 0;
};

class MPCCache_ocs {
   public:
    explicit MPCCache_ocs(size_t capacity = 32, double default_delta = 1.0) //so that implicit conversion doesn't happen, puch hi lo
        : capacity_(capacity), default_delta_(default_delta) {}

    void setBinWidths(const std::vector<double> &h) {
        std::lock_guard<std::mutex> lg(m_); //why mutex here?
        h_ = h;
    }

    const std::vector<double> &getBinWidths() const { return h_; } 

    /*
    * @brief Quantizes the feature vector into a string key.
    */
    std::string quantizeKey(const ocs2::vector_t &feat) const { 
        std::string s;
        s.reserve(feat.size() * 4);
        for (size_t i = 0; i < feat.size(); ++i) {
            double denom = (h_.size() == feat.size() ? h_[i] : 0.5);
            long long qi = llround(feat[i] / denom); //llround rounds a floating point number to the nearest long long, a simple cast truncates instead of rouding
            s += std::to_string(qi);
            s.push_back(',');
        }
        return s;
    }

    /*
    * @brief Inserts an entry into the cache.
        @param [in] qkey: quantized key
        @param [in] entry: the cache entry to be inserted
    */
    void insert(const std::string &qkey, MPCCacheEntry_ocs &entry) {  
        std::lock_guard<std::mutex> lg(m_);
        entry.last_use = ++tick_;
        map_[qkey] = std::move(entry);
        if (map_.size() > capacity_) evictLRU();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lg(m_);
        return map_.size();
    }

    /*
    *@brief Queries the nearest entry in the cache within a given distance.
    */
    MPCCacheEntry_ocs *queryNearest(const std::string &qkey,
                                const ocs2::vector_t &feat, double delta) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = map_.find(qkey);
        if (it == map_.end())   return nullptr;
        MPCCacheEntry_ocs &e = it->second;
        double d = euclidean(feat, e.feat);
        // std::cerr<< "dist :" << d <<std::endl;
        if (d <= delta ) {
            e.last_use = ++tick_;
            return &e;
        }
        return nullptr;
    }

    void clear() {
        std::lock_guard<std::mutex> lg(m_);
        map_.clear();
    }

    void save(std::ostream& os) const {
        std::lock_guard<std::mutex> lg(m_);
        // Write capacity_
        os.write(reinterpret_cast<const char*>(&capacity_), sizeof(size_t));
        // Write default_delta_
        os.write(reinterpret_cast<const char*>(&default_delta_),
                sizeof(double));
        // Write tick_
        os.write(reinterpret_cast<const char*>(&tick_), sizeof(uint64_t));
        // Write h_
        size_t h_size = h_.size();
        os.write(reinterpret_cast<const char*>(&h_size), sizeof(size_t));
        for (double d : h_) {
            os.write(reinterpret_cast<const char*>(&d), sizeof(double));
        }
        // Write map_.size()
        size_t map_size = map_.size();
        os.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
        // Write each entry
        for (const auto& p : map_) {
            const MPCCacheEntry_ocs& e = p.second;
            // feat
            size_t feat_size = e.feat.size();
            os.write(reinterpret_cast<const char*>(&feat_size), sizeof(size_t));
            for (size_t i = 0; i < feat_size; ++i) {
                double val = static_cast<double>(e.feat[i]);
                os.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }

            // primalSolution: timeTrajectory_
            const auto &pt = e.ptr.primalSolution.timeTrajectory_;
            size_t pt_size = pt.size();
            os.write(reinterpret_cast<const char*>(&pt_size), sizeof(size_t));
            for (double t : pt) os.write(reinterpret_cast<const char*>(&t), sizeof(double));

            // stateTrajectory_
            const auto &st = e.ptr.primalSolution.stateTrajectory_;
            size_t st_size = st.size();
            os.write(reinterpret_cast<const char*>(&st_size), sizeof(size_t));
            for (const auto &v : st) {
                size_t vec_size = v.size();
                os.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
                for (size_t i = 0; i < vec_size; ++i) {
                    double val = static_cast<double>(v[i]);
                    os.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }

            // inputTrajectory_
            const auto &it = e.ptr.primalSolution.inputTrajectory_;
            size_t it_size = it.size();
            os.write(reinterpret_cast<const char*>(&it_size), sizeof(size_t));
            for (const auto &v : it) {
                size_t vec_size = v.size();
                os.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
                for (size_t i = 0; i < vec_size; ++i) {
                    double val = static_cast<double>(v[i]);
                    os.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }

            // postEventIndices_
            const auto &pei = e.ptr.primalSolution.postEventIndices_;
            size_t pei_size = pei.size();
            os.write(reinterpret_cast<const char*>(&pei_size), sizeof(size_t));
            for (size_t idx : pei) os.write(reinterpret_cast<const char*>(&idx), sizeof(size_t));

            // ModeSchedule
            const auto &ms = e.ptr.primalSolution.modeSchedule_;
            size_t et_size = ms.eventTimes.size();
            os.write(reinterpret_cast<const char*>(&et_size), sizeof(size_t));
            for (double et : ms.eventTimes) os.write(reinterpret_cast<const char*>(&et), sizeof(double));
            size_t ms_size = ms.modeSequence.size();
            os.write(reinterpret_cast<const char*>(&ms_size), sizeof(size_t));
            for (size_t m : ms.modeSequence) os.write(reinterpret_cast<const char*>(&m), sizeof(size_t));

            // PerformanceIndex (8 doubles)
            const auto &pi = e.ptr.performanceIndices;
            double vals[8] = {pi.merit, pi.cost, pi.dualFeasibilitiesSSE, pi.dynamicsViolationSSE,
                              pi.equalityConstraintsSSE, pi.inequalityConstraintsSSE,
                              pi.equalityLagrangian, pi.inequalityLagrangian};
            os.write(reinterpret_cast<const char*>(&vals), sizeof(vals));

            // CommandData: SystemObservation
            const auto &obs = e.ptr.command.mpcInitObservation_;
            os.write(reinterpret_cast<const char*>(&obs.mode), sizeof(size_t));
            os.write(reinterpret_cast<const char*>(&obs.time), sizeof(double));
            size_t obs_state_size = obs.state.size();
            os.write(reinterpret_cast<const char*>(&obs_state_size), sizeof(size_t));
            for (size_t i = 0; i < obs_state_size; ++i) {
                double val = static_cast<double>(obs.state[i]);
                os.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
            size_t obs_input_size = obs.input.size();
            os.write(reinterpret_cast<const char*>(&obs_input_size), sizeof(size_t));
            for (size_t i = 0; i < obs_input_size; ++i) {
                double val = static_cast<double>(obs.input[i]);
                os.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }

            // CommandData: TargetTrajectories
            const auto &tt = e.ptr.command.mpcTargetTrajectories_;
            size_t tt_time_size = tt.timeTrajectory.size();
            os.write(reinterpret_cast<const char*>(&tt_time_size), sizeof(size_t));
            for (double t : tt.timeTrajectory) os.write(reinterpret_cast<const char*>(&t), sizeof(double));
            size_t tt_state_size = tt.stateTrajectory.size();
            os.write(reinterpret_cast<const char*>(&tt_state_size), sizeof(size_t));
            for (const auto &v : tt.stateTrajectory) {
                size_t vec_size = v.size();
                os.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
                for (size_t i = 0; i < vec_size; ++i) {
                    double val = static_cast<double>(v[i]);
                    os.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }
            size_t tt_input_size = tt.inputTrajectory.size();
            os.write(reinterpret_cast<const char*>(&tt_input_size), sizeof(size_t));
            for (const auto &v : tt.inputTrajectory) {
                size_t vec_size = v.size();
                os.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
                for (size_t i = 0; i < vec_size; ++i) {
                    double val = static_cast<double>(v[i]);
                    os.write(reinterpret_cast<const char*>(&val), sizeof(double));
                }
            }

            // trust radius and last use
            os.write(reinterpret_cast<const char*>(&e.trust_radius), sizeof(double));
            os.write(reinterpret_cast<const char*>(&e.last_use), sizeof(uint64_t));
        }
    }

    void load(std::istream& is) {
        std::lock_guard<std::mutex> lg(m_);
        map_.clear();
        // Read capacity_
        is.read(reinterpret_cast<char*>(&capacity_), sizeof(size_t));
        // Read default_delta_
        is.read(reinterpret_cast<char*>(&default_delta_), sizeof(double));
        // Read tick_
        is.read(reinterpret_cast<char*>(&tick_), sizeof(uint64_t));
        // Read h_
        size_t h_size;
        is.read(reinterpret_cast<char*>(&h_size), sizeof(size_t));
        h_.resize(h_size);
        for (size_t i = 0; i < h_size; ++i) {
            is.read(reinterpret_cast<char*>(&h_[i]), sizeof(double));
        }
        // Read map_size
        size_t map_size;
        is.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
        // Read each entry
        for (size_t j = 0; j < map_size; ++j) {
            MPCCacheEntry_ocs e;
            // feat
            size_t feat_size;
            is.read(reinterpret_cast<char*>(&feat_size), sizeof(size_t));
            e.feat.resize(feat_size);
            for (size_t i = 0; i < feat_size; ++i) {
                double val;
                is.read(reinterpret_cast<char*>(&val), sizeof(double));
                e.feat[i] = val;
            }

            // primalSolution: timeTrajectory_
            size_t pt_size;
            is.read(reinterpret_cast<char*>(&pt_size), sizeof(size_t));
            e.ptr.primalSolution.timeTrajectory_.resize(pt_size);
            for (size_t i = 0; i < pt_size; ++i) {
                double t;
                is.read(reinterpret_cast<char*>(&t), sizeof(double));
                e.ptr.primalSolution.timeTrajectory_[i] = t;
            }

            // stateTrajectory_
            size_t st_size;
            is.read(reinterpret_cast<char*>(&st_size), sizeof(size_t));
            e.ptr.primalSolution.stateTrajectory_.resize(st_size);
            for (size_t k = 0; k < st_size; ++k) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.primalSolution.stateTrajectory_[k] = v;
            }

            // inputTrajectory_
            size_t it_size;
            is.read(reinterpret_cast<char*>(&it_size), sizeof(size_t));
            e.ptr.primalSolution.inputTrajectory_.resize(it_size);
            for (size_t k = 0; k < it_size; ++k) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.primalSolution.inputTrajectory_[k] = v;
            }

            // postEventIndices_
            size_t pei_size;
            is.read(reinterpret_cast<char*>(&pei_size), sizeof(size_t));
            e.ptr.primalSolution.postEventIndices_.resize(pei_size);
            for (size_t i = 0; i < pei_size; ++i) {
                size_t idx;
                is.read(reinterpret_cast<char*>(&idx), sizeof(size_t));
                e.ptr.primalSolution.postEventIndices_[i] = idx;
            }

            // ModeSchedule
            size_t et_size;
            is.read(reinterpret_cast<char*>(&et_size), sizeof(size_t));
            e.ptr.primalSolution.modeSchedule_.eventTimes.resize(et_size);
            for (size_t i = 0; i < et_size; ++i) {
                double et;
                is.read(reinterpret_cast<char*>(&et), sizeof(double));
                e.ptr.primalSolution.modeSchedule_.eventTimes[i] = et;
            }
            size_t ms_size;
            is.read(reinterpret_cast<char*>(&ms_size), sizeof(size_t));
            e.ptr.primalSolution.modeSchedule_.modeSequence.resize(ms_size);
            for (size_t i = 0; i < ms_size; ++i) {
                size_t m;
                is.read(reinterpret_cast<char*>(&m), sizeof(size_t));
                e.ptr.primalSolution.modeSchedule_.modeSequence[i] = m;
            }

            // PerformanceIndex
            double vals[8];
            is.read(reinterpret_cast<char*>(&vals), sizeof(vals));
            e.ptr.performanceIndices.merit = vals[0];
            e.ptr.performanceIndices.cost = vals[1];
            e.ptr.performanceIndices.dualFeasibilitiesSSE = vals[2];
            e.ptr.performanceIndices.dynamicsViolationSSE = vals[3];
            e.ptr.performanceIndices.equalityConstraintsSSE = vals[4];
            e.ptr.performanceIndices.inequalityConstraintsSSE = vals[5];
            e.ptr.performanceIndices.equalityLagrangian = vals[6];
            e.ptr.performanceIndices.inequalityLagrangian = vals[7];

            // CommandData: SystemObservation
            is.read(reinterpret_cast<char*>(&e.ptr.command.mpcInitObservation_.mode), sizeof(size_t));
            is.read(reinterpret_cast<char*>(&e.ptr.command.mpcInitObservation_.time), sizeof(double));
            size_t obs_state_size;
            is.read(reinterpret_cast<char*>(&obs_state_size), sizeof(size_t));
            e.ptr.command.mpcInitObservation_.state.resize(obs_state_size);
            for (size_t i = 0; i < obs_state_size; ++i) {
                double val;
                is.read(reinterpret_cast<char*>(&val), sizeof(double));
                e.ptr.command.mpcInitObservation_.state[i] = val;
            }
            size_t obs_input_size;
            is.read(reinterpret_cast<char*>(&obs_input_size), sizeof(size_t));
            e.ptr.command.mpcInitObservation_.input.resize(obs_input_size);
            for (size_t i = 0; i < obs_input_size; ++i) {
                double val;
                is.read(reinterpret_cast<char*>(&val), sizeof(double));
                e.ptr.command.mpcInitObservation_.input[i] = val;
            }

            // CommandData: TargetTrajectories
            size_t tt_time_size;
            is.read(reinterpret_cast<char*>(&tt_time_size), sizeof(size_t));
            e.ptr.command.mpcTargetTrajectories_.timeTrajectory.resize(tt_time_size);
            for (size_t i = 0; i < tt_time_size; ++i) {
                double t;
                is.read(reinterpret_cast<char*>(&t), sizeof(double));
                e.ptr.command.mpcTargetTrajectories_.timeTrajectory[i] = t;
            }
            size_t tt_state_size;
            is.read(reinterpret_cast<char*>(&tt_state_size), sizeof(size_t));
            e.ptr.command.mpcTargetTrajectories_.stateTrajectory.resize(tt_state_size);
            for (size_t k = 0; k < tt_state_size; ++k) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.command.mpcTargetTrajectories_.stateTrajectory[k] = v;
            }
            size_t tt_input_size;
            is.read(reinterpret_cast<char*>(&tt_input_size), sizeof(size_t));
            e.ptr.command.mpcTargetTrajectories_.inputTrajectory.resize(tt_input_size);
            for (size_t k = 0; k < tt_input_size; ++k) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.command.mpcTargetTrajectories_.inputTrajectory[k] = v;
            }

            // trust radius and last use
            is.read(reinterpret_cast<char*>(&e.trust_radius), sizeof(double));
            is.read(reinterpret_cast<char*>(&e.last_use), sizeof(uint64_t));

            // Compute key and insert directly (preserve last_use)
            std::string key = quantizeKey(e.feat);
            map_[key] = std::move(e);
        }
        // Safety: ensure capacity invariant even if file was manually edited or
        // saved with bug
        while (map_.size() > capacity_) {
            evictLRU();
        }
    }

   private:

    static double euclidean(const ocs2::vector_t &a,
                            const ocs2::vector_t &b) {
        if (a.size() != b.size())
            return std::numeric_limits<double>::infinity(); //returns value as inf, why not just use INT_MAX?
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }

    /*
    *@brief Evicts the least recently used entry from the cache.
    */
    void evictLRU() {
        uint64_t oldest = UINT64_MAX;
        std::string oldest_key;
        for (auto &p : map_) {
            if (p.second.last_use < oldest) {
                oldest = p.second.last_use;
                oldest_key = p.first;
            }
        }
        if (!oldest_key.empty()) map_.erase(oldest_key);
    }
    

    std::unordered_map<std::string, MPCCacheEntry_ocs> map_;
    std::vector<double> h_; //bin
    size_t capacity_;
    double default_delta_;
    mutable std::mutex m_; //why use mutable with mutex?
    uint64_t tick_ = 0;
};

// Free functions to save/load a vector<unique_ptr<MPCCache>> to/from a single
// file
inline bool save_caches(const std::vector<std::unique_ptr<MPCCache_ocs>>& caches,
                        const std::string& filename) {
    std::ofstream os(filename, std::ios::binary);
    if (!os) return false;  // Error handling omitted for simplicity
    size_t sz = caches.size();
    os.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
    for (const auto& cache : caches) {
        cache->save(os);
    }

    return true;
}

inline bool load_caches(std::vector<std::unique_ptr<MPCCache_ocs>>& caches,
                        const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);
    if (!is) return false;  // Error handling omitted for simplicity
    size_t sz;
    is.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
    caches.clear();
    caches.reserve(sz);
    for (size_t i = 0; i < sz; ++i) {
        auto cache = std::make_unique<MPCCache_ocs>();
        cache->load(is);
        caches.push_back(std::move(cache));
        std::cerr<<"loading cache: " << i << " st \n";
    }

    return true;
}