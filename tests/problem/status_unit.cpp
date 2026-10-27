// Unit tests for ocp::to_string(Status) (include/ocp/problem.hpp):
// every enumerator maps to its own spelling, and an out-of-range
// value maps to "unknown".

#include <cstdio>
#include <string>

#include "ocp/problem.hpp"

namespace
{

using ocp::Status;
using ocp::to_string;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

void test_enum_spelling()
{
    check(std::string(to_string(Status::kUnset)) == "kUnset", "kUnset");
    check(std::string(to_string(Status::kSolved)) == "kSolved", "kSolved");
    check(std::string(to_string(Status::kMaxIterations)) == "kMaxIterations",
          "kMaxIterations");
    check(std::string(to_string(Status::kInfeasible)) == "kInfeasible",
          "kInfeasible");
    check(std::string(to_string(Status::kQpFailure)) == "kQpFailure",
          "kQpFailure");
    check(std::string(to_string(Status::kMinStep)) == "kMinStep", "kMinStep");
    check(std::string(to_string(Status::kUnbounded)) == "kUnbounded",
          "kUnbounded");
    check(std::string(to_string(Status::kNanDetected)) == "kNanDetected",
          "kNanDetected");
    check(std::string(to_string(Status::kAborted)) == "kAborted", "kAborted");
    check(std::string(to_string(Status::kTimeout)) == "kTimeout",
          "kTimeout");
    check(std::string(to_string(Status::kNotInitialized)) ==
              "kNotInitialized",
          "kNotInitialized");
}

void test_out_of_range()
{
    check(std::string(to_string(static_cast<Status>(99))) == "unknown",
          "out-of-range maps to unknown");
    check(std::string(to_string(static_cast<Status>(-1))) == "unknown",
          "negative maps to unknown");
}

}  // namespace

int main()
{
    test_enum_spelling();
    test_out_of_range();

    if (failures == 0)
    {
        std::printf("All status_name checks passed.\n");
    }
    return failures;
}
