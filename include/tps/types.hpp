#pragma once
#include <string>
#include <vector>

namespace tps {

struct ResidualResult {
    std::string verdict;      // "good" | "bad" | "unknown"
    std::string reason;       // set when verdict == "unknown"
    std::string selection_note;

    double rmse          = 0.0;
    int    n             = 0;
    int    runs          = 0;
    double runs_expected = 0.0;
    bool   runs_ok       = true;
    double durbin_watson = 2.0;
    bool   dw_ok         = true;
    bool   hetero_ok     = true;
    bool   trend_ok      = true;
    std::vector<std::string> issues;
};

struct BestWindowResult {
    int           start  = 0;
    int           end    = 0;
    ResidualResult result;
};

}  // namespace tps
