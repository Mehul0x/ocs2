#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <ocs2_core/Types.h>
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/MRT_BASE.h"

struct MPCSolutionPointers {
    std::unique_ptr<ocs2::PrimalSolution> primalSolutionPtr;
    std::unique_ptr<ocs2::CommandData> commandPtr;
    std::unique_ptr<ocs2::PerformanceIndex> performanceIndicesPtr;

    // This is a convenient helper to check if the result is valid.
    explicit operator bool() const {
        return primalSolutionPtr != nullptr;
    }
};

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

    std::string quantizeKey(const ocs2::vector_t &feat) const { 
        std::string s;
        s.reserve(feat.size() * 4);
        for (size_t i = 0; i < feat.size(); ++i) {
            double denom = (h_.size() == feat.size() ? h_[i] : 1.0);
            long long qi = llround(feat[i] / denom); //llround rounds a floating point number to the nearest long long, a simple cast truncates instead of rouding
            s += std::to_string(qi);
            s.push_back(',');
        }
        return s;
    }

    void insert(const std::string &qkey, MPCCacheEntry_ocs &entry) {  //yahan pe pehle &&entry tha , why?
        std::lock_guard<std::mutex> lg(m_);
        entry.last_use = ++tick_;
        map_[qkey] = std::move(entry);
        if (map_.size() > capacity_) evictLRU();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lg(m_);
        return map_.size();
    }

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