/*
Github@Mehul0x

1. Create a hash function (a(random vector), b(random number), w(bin width) )
2. Concatonate k hashes -> signature
3. Maintain L independent tables of such signatures
4. Query process-> 
    a. Compute L signatures for the query point
    b. For each signature, retrieve points from the corresponding table bucket
    c. Compute true Euclidean distance to verify
    d. If any distance ≤ delta → use cached solution

Tunable params: 
    1. k = number of hashes per signature (higher k -> fewer collisions -> fewer candidates)
    2. L = number of hash tables (higher L -> more candidates -> higher recall)
    3. w = bin width (larger w -> more collisions -> more candidates)

We fix them as w = 1.4* delta, k= 5 , L is chose such that the recall >0.95 = 35 

instead of strings, we are using bits

tableId -> id for the L independent hash tables

radius search currently searches for the nearest neighbour instead of just returning the first
neighbour within a delta-> more stable ig
*/

#include <vector>
#include <string>
#include <limits>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <ocs2_core/Types.h>
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/MRT_BASE.h"

using namespace std;
namespace ocs2
{

struct MPCSolution {
    PrimalSolution primalSolution;
    CommandData command;
    PerformanceIndex performanceIndices;
};

struct MPCLSHEntry{
    vector_t feat;
    MPCSolution ptr;
};

/*
There will be L HashTables

Packed LSH signature -> uint64_t
indices of all features that hashed into that bucket -> vector<int>
*/
struct HashTable{
    std::unordered_map<uint64_t, vector<int>> table;
};
    



// FIXED SEED → deterministic hash functions
mt19937 rng = mt19937(69);
class LSH {
    public:
        LSH(double approximate_radius=0.1 , int dim=10, int num_hashes=5, int num_tables=35): 
                delta(approximate_radius), dim(dim), k(num_hashes), L(num_tables){
            
            w=1.4* approximate_radius;
            
            //since we need a for each table and each hash in it
            a.resize(L, vector<vector<double>>(k, vector<double>(dim)));

            //we also need b for each table and each hash in it
            b.resize(L, vector<double>(k));
            tables.resize(L);

            normal_distribution<double> gauss(0.0, 1.0);
            uniform_real_distribution<double> uni(0.0, w);
            
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < k; j++) {
                    for (int d = 0; d < dim; d++)
                        a[i][j][d] = gauss(rng);
                    b[i][j] = uni(rng);
                }
            }
        }


        

        /*
        
        */
        void insert(const vector_t& feat, const MPCLSHEntry& entry)
        {
            int idx = entries.size();

            entries.push_back(entry);         // store full entry
            points.push_back(EigenToStd(feat)); // store numeric feature for LSH hashing

            // Insert point index into each hash table
            for (int t = 0; t < L; t++) {
                uint64_t key = hashSignature(t, points[idx]);
                tables[t].table[key].push_back(idx);
            }
        }
        
        
        /*
        @brief: Returns the solution after peroforming a neighbour search

        @in: q = query, we want to find
        */
        MPCLSHEntry* queryNearest(const vector_t& feat) {
            vector<double> q = EigenToStd(feat);
            int idx = nearestSearch(q, delta);

            return (idx == -1 ? nullptr : &entries[idx]);
        }
        // Persist this LSH instance (points, entries, tables). Does NOT save a & b.
        bool saveToStream(std::ostream& os) const;

        // Load an LSH instance from stream. Returns nullptr on failure.
        static std::unique_ptr<LSH> loadFromStream(std::istream& is);
    private:


        /*
        @brief: pack k hash integers into a 64 bit key
        
        @in: k integers after getting hashed
        */
        inline uint64_t packKey(const vector<int>& h) const{
            uint64_t key =0;
            for(int i=0; i<k;i++){
                uint64_t v = (uint64_t)(h[i] & 0xFFFULL); //12 bit-signed wrap
                key |= (v << (12*i));
            }
            return key;
        }

        /*
        @in -> vector being hashed
        @out -> 64 bit hash signature
        */
        uint64_t hashSignature(int tableId, const vector<double>& v) const {
        vector<int> h(k);

        for (int j = 0; j < k; j++) {
            double dot = 0.0;
            for (int d = 0; d < dim; d++)
                dot += a[tableId][j][d] * v[d];

            h[j] = (int)floor((dot + b[tableId][j]) / w);
        }
        return packKey(h);
    }


        /*
        @brief: Computes the hash key for each table and then looks up that bucket in hash table t

        Returns the nearest neighbour, I might be loosing time in this?
        @in: q= query vector , Rq = Distance threshold
        @out: list of indices into the points array inside the LSH
        */

        int nearestSearch(const vector<double>& q, double Rq) const {
            unordered_set<int> cand;
            cand.reserve(256);

            double R2 = Rq * Rq;

            for (int t = 0; t < L; t++) {
                uint64_t key = hashSignature(t, q);
                auto it = tables[t].table.find(key);
                if (it == tables[t].table.end()) continue;

                for (int idx : it->second)
                    cand.insert(idx);
            }

            int bestIdx = -1;
            double bestDist = 1e100;

            for (int idx : cand) {
                double d2 = dist2(points[idx], q);

                if (d2 <= R2 && d2 < bestDist) {
                    bestDist = d2;
                    bestIdx = idx;
                }
                
            }

            return bestIdx;
        }

        /*
        returns the first neighbour with validated delta
        */
        int firstvalidatedNeighborSearch(const vector<double>& q, double Rq) const {
            double R2 = Rq * Rq;

            // Check tables one by one
            for (int t = 0; t < L; t++) {
                uint64_t key = hashSignature(t, q);

                auto it = tables[t].table.find(key);
                if (it == tables[t].table.end())
                    continue;

                // Check candidates IN THIS TABLE ONLY
                for (int idx : it->second) {
                    // Exact verification
                    if (dist2(points[idx], q) <= R2)
                        return idx;   // FIRST match → return immediately
                }
            }

            return -1; // no neighbor found
        }

        int firstunvalidatedNeigborSearch(const vector<double>& q, double Rq) const{
            // Check tables one by one
            for (int t = 0; t < L; t++) {
            uint64_t key = hashSignature(t, q);

            auto it = tables[t].table.find(key);
            if (it == tables[t].table.end())
                continue;

            // Return the FIRST entry in the bucket with NO dist check
            if (!it->second.empty())
                return it->second[0];
            }

            // No bucket had anything
            return -1;
            
        }
        


        inline double dist2(const vector<double>& a, const vector<double>& b) const {
            double s = 0;
            for (int i = 0; i < dim; i++) {
                double d = a[i] - b[i];
                s += d * d;
            }
            return s;
        }


        //Converts ocs2::vector_t to std::vector<double>
        inline std::vector<double> EigenToStd(const vector_t& eigen_vec) {
            int size = eigen_vec.size();
            std::vector<double> std_vec(size);
            std::copy(eigen_vec.data(), eigen_vec.data() + size, std_vec.begin());
            return std_vec;
        }

        //converts std::vector<double> to ocs2::vector_t
        inline vector_t StdToEigen(const std::vector<double>& std_vec) {
            int size = static_cast<int>(std_vec.size());
            // Create a map/view of the std::vector's data
            Eigen::Map<const vector_t> map_vec(std_vec.data(), size);
            // Copy the map's contents into a new, distinct Eigen vector
            return map_vec;
        }

        
        double w, delta;
        int k, L, dim;
        
        vector<HashTable> tables;

        vector<vector<double>> points;    // raw double features
        vector<MPCLSHEntry> entries;      // FULL cached entries



        vector<vector<vector<double>>> a; // L x k x dim
        vector<vector<double>> b;         // L x k
    };

    // Member save/load implementations
    inline bool LSH::saveToStream(std::ostream& os) const {
        if (!os) return false;
        // header
        os.write(reinterpret_cast<const char*>(&delta), sizeof(double));
        os.write(reinterpret_cast<const char*>(&dim), sizeof(int));
        os.write(reinterpret_cast<const char*>(&k), sizeof(int));
        os.write(reinterpret_cast<const char*>(&L), sizeof(int));
        os.write(reinterpret_cast<const char*>(&w), sizeof(double));

        // points
        size_t points_size = points.size();
        os.write(reinterpret_cast<const char*>(&points_size), sizeof(size_t));
        for (const auto& p : points) {
            size_t psize = p.size();
            os.write(reinterpret_cast<const char*>(&psize), sizeof(size_t));
            for (double v : p) os.write(reinterpret_cast<const char*>(&v), sizeof(double));
        }

        // entries
        size_t entries_size = entries.size();
        os.write(reinterpret_cast<const char*>(&entries_size), sizeof(size_t));
        for (const auto& e : entries) {
            // feat
            size_t feat_size = e.feat.size();
            os.write(reinterpret_cast<const char*>(&feat_size), sizeof(size_t));
            for (int i = 0; i < feat_size; ++i) {
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
        }

        // tables
        size_t tables_size = tables.size();
        os.write(reinterpret_cast<const char*>(&tables_size), sizeof(size_t));
        for (const auto &ht : tables) {
            size_t map_size = ht.table.size();
            os.write(reinterpret_cast<const char*>(&map_size), sizeof(size_t));
            for (const auto &p : ht.table) {
                uint64_t key = p.first;
                os.write(reinterpret_cast<const char*>(&key), sizeof(uint64_t));
                size_t vec_size = p.second.size();
                os.write(reinterpret_cast<const char*>(&vec_size), sizeof(size_t));
                for (int idx : p.second) {
                    int32_t v = static_cast<int32_t>(idx);
                    os.write(reinterpret_cast<const char*>(&v), sizeof(int32_t));
                }
            }
        }
        return os.good();
    }

    inline std::unique_ptr<LSH> LSH::loadFromStream(std::istream& is) {
        if (!is) return nullptr;
        double delta;
        int dim_, k_, L_;
        double w_;
        is.read(reinterpret_cast<char*>(&delta), sizeof(double));
        if (!is) return nullptr;
        is.read(reinterpret_cast<char*>(&dim_), sizeof(int));
        is.read(reinterpret_cast<char*>(&k_), sizeof(int));
        is.read(reinterpret_cast<char*>(&L_), sizeof(int));
        is.read(reinterpret_cast<char*>(&w_), sizeof(double));

        auto cache = std::make_unique<LSH>(delta, dim_, k_, L_);

        // points
        size_t points_size;
        is.read(reinterpret_cast<char*>(&points_size), sizeof(size_t));
        cache->points.clear();
        cache->points.reserve(points_size);
        for (size_t pi = 0; pi < points_size; ++pi) {
            size_t psize;
            is.read(reinterpret_cast<char*>(&psize), sizeof(size_t));
            std::vector<double> p(psize);
            for (size_t i = 0; i < psize; ++i) is.read(reinterpret_cast<char*>(&p[i]), sizeof(double));
            cache->points.push_back(std::move(p));
        }

        // entries
        size_t entries_size;
        is.read(reinterpret_cast<char*>(&entries_size), sizeof(size_t));
        cache->entries.clear();
        cache->entries.reserve(entries_size);
        for (size_t ei = 0; ei < entries_size; ++ei) {
            MPCLSHEntry e;
            size_t feat_size;
            is.read(reinterpret_cast<char*>(&feat_size), sizeof(size_t));
            std::vector<double> feat_std(feat_size);
            for (size_t i = 0; i < feat_size; ++i) is.read(reinterpret_cast<char*>(&feat_std[i]), sizeof(double));
            // convert to vector_t
            ocs2::vector_t fv(feat_size);
            for (size_t i = 0; i < feat_size; ++i) fv[i] = feat_std[i];
            e.feat = fv;

            // primalSolution: timeTrajectory_
            size_t pt_size;
            is.read(reinterpret_cast<char*>(&pt_size), sizeof(size_t));
            e.ptr.primalSolution.timeTrajectory_.resize(pt_size);
            for (size_t i = 0; i < pt_size; ++i) is.read(reinterpret_cast<char*>(&e.ptr.primalSolution.timeTrajectory_[i]), sizeof(double));

            // stateTrajectory_
            size_t st_size;
            is.read(reinterpret_cast<char*>(&st_size), sizeof(size_t));
            e.ptr.primalSolution.stateTrajectory_.resize(st_size);
            for (size_t kidx = 0; kidx < st_size; ++kidx) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.primalSolution.stateTrajectory_[kidx] = v;
            }

            // inputTrajectory_
            size_t it_size;
            is.read(reinterpret_cast<char*>(&it_size), sizeof(size_t));
            e.ptr.primalSolution.inputTrajectory_.resize(it_size);
            for (size_t kidx = 0; kidx < it_size; ++kidx) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.primalSolution.inputTrajectory_[kidx] = v;
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
            for (size_t kidx = 0; kidx < tt_state_size; ++kidx) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.command.mpcTargetTrajectories_.stateTrajectory[kidx] = v;
            }
            size_t tt_input_size;
            is.read(reinterpret_cast<char*>(&tt_input_size), sizeof(size_t));
            e.ptr.command.mpcTargetTrajectories_.inputTrajectory.resize(tt_input_size);
            for (size_t kidx = 0; kidx < tt_input_size; ++kidx) {
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                ocs2::vector_t v(vec_size);
                for (size_t i = 0; i < vec_size; ++i) {
                    double val;
                    is.read(reinterpret_cast<char*>(&val), sizeof(double));
                    v[i] = val;
                }
                e.ptr.command.mpcTargetTrajectories_.inputTrajectory[kidx] = v;
            }

            cache->entries.push_back(std::move(e));
        }

        // tables
        size_t tables_size;
        is.read(reinterpret_cast<char*>(&tables_size), sizeof(size_t));
        cache->tables.clear();
        cache->tables.resize(tables_size);
        for (size_t ti = 0; ti < tables_size; ++ti) {
            size_t map_size;
            is.read(reinterpret_cast<char*>(&map_size), sizeof(size_t));
            for (size_t mi = 0; mi < map_size; ++mi) {
                uint64_t key;
                is.read(reinterpret_cast<char*>(&key), sizeof(uint64_t));
                size_t vec_size;
                is.read(reinterpret_cast<char*>(&vec_size), sizeof(size_t));
                std::vector<int> idxs;
                idxs.reserve(vec_size);
                for (size_t ii = 0; ii < vec_size; ++ii) {
                    int32_t v;
                    is.read(reinterpret_cast<char*>(&v), sizeof(int32_t));
                    idxs.push_back(static_cast<int>(v));
                }
                cache->tables[ti].table[key] = std::move(idxs);
            }
        }

        return cache;
    }
    // Serialize a vector of LSH caches to a file. Uses member saveToStream.
    inline bool save_caches(const std::vector<std::unique_ptr<LSH>>& caches, const std::string& filename) {
        std::ofstream os(filename, std::ios::binary);
        if (!os) return false;
        size_t sz = caches.size();
        os.write(reinterpret_cast<const char*>(&sz), sizeof(size_t));
        for (const auto& cache : caches) {
            bool ok = true;
            if (!cache) {
                // write false marker
                uint8_t present = 0;
                os.write(reinterpret_cast<const char*>(&present), sizeof(uint8_t));
                continue;
            }
            uint8_t present = 1;
            os.write(reinterpret_cast<const char*>(&present), sizeof(uint8_t));
            ok = cache->saveToStream(os);
            if (!ok) return false;
        }
        return true;
    }

    inline bool load_caches(std::vector<std::unique_ptr<LSH>>& caches, const std::string& filename) {
        std::ifstream is(filename, std::ios::binary);
        if (!is) return false;
        size_t sz;
        is.read(reinterpret_cast<char*>(&sz), sizeof(size_t));
        caches.clear();
        caches.reserve(sz);
        for (size_t ci = 0; ci < sz; ++ci) {
            uint8_t present = 0;
            is.read(reinterpret_cast<char*>(&present), sizeof(uint8_t));
            if (!is) return false;
            if (!present) {
                caches.push_back(nullptr);
                continue;
            }
            auto cache = LSH::loadFromStream(is);
            if (!cache) return false;
            caches.push_back(std::move(cache));
        }
        return true;
    }


}//namespace ocs2