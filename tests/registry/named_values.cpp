// ValueRegistry unit tests (docs/plans/DUAL_ACCESS_PLAN.md, step 1):
// add / get / set / has / names / dump, type mismatch, unknown names,
// dotted delegation (incl. a two-level chain), the add() range guard,
// and offset / copy / move / assignment semantics with nested registries.

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "ocp/named_values.hpp"

namespace
{

using ocp::ValueRegistry;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

struct Inner
{
    double a = 1.5;
    int b = 7;
    bool c = true;
    ValueRegistry reg;
    Inner()
    {
        reg.begin(*this);
        reg.add("a", a);
        reg.add("b", b);
        reg.add("c", c);
    }
};

struct Outer
{
    double x = 1.0;
    int y = 2;
    bool z = false;
    Inner inner;
    ValueRegistry reg;
    Outer()
    {
        reg.begin(*this);
        reg.add("x", x);
        reg.add("y", y);
        reg.add("z", z);
        reg.add("inner", inner.reg);
    }
};

struct Deep
{
    double v = 3.25;
    ValueRegistry reg;
    Deep()
    {
        reg.begin(*this);
        reg.add("v", v);
    }
};

struct Mid
{
    int n = 4;
    Deep deep;
    ValueRegistry reg;
    Mid()
    {
        reg.begin(*this);
        reg.add("n", n);
        reg.add("deep", deep.reg);
    }
};

struct Top
{
    double d = 5.5;
    Mid mid;
    ValueRegistry reg;
    Top()
    {
        reg.begin(*this);
        reg.add("d", d);
        reg.add("mid", mid.reg);
    }
};

void test_scalar_access()
{
    Outer o;

    check(o.reg.size() == 4, "size: 3 scalars + 1 registry link");
    check(o.inner.reg.size() == 3, "nested size: 3 scalars");

    double xd = 0.0;
    int yi = 0;
    bool zb = true;
    check(o.reg.get("x", xd) && xd == 1.0, "get x");
    check(o.reg.get("y", yi) && yi == 2, "get y");
    check(o.reg.get("z", zb) && zb == false, "get z");

    check(o.reg.set("x", 2.5), "set x");
    check(o.x == 2.5, "set x wrote the member");
    check(o.reg.set("y", 42), "set y");
    check(o.y == 42, "set y wrote the member");
    check(o.reg.set("z", true), "set z");
    check(o.z, "set z wrote the member");

    check(o.reg.has("x"), "has x");
    check(o.reg.has("y"), "has y");
    check(o.reg.has("inner.a"), "has dotted name");
    check(!o.reg.has("nope"), "has unknown name");
    check(!o.reg.get("nope", xd), "get unknown name");
    check(!o.reg.set("nope", 1.0), "set unknown name");
}

void test_type_mismatch()
{
    Outer o;
    double d = 0.0;
    int i = 0;
    bool b = false;
    check(!o.reg.get("y", d), "get<int-entry> as double");
    check(!o.reg.set("y", 1.5), "set<int-entry> with double");
    check(!o.reg.get("x", i), "get<double-entry> as int");
    check(!o.reg.set("x", 1), "set<double-entry> with int");
    check(!o.reg.get("z", d), "get<bool-entry> as double");
    check(!o.reg.get("inner.a", b), "get dotted with wrong type");
    // failed lookups leave the out value untouched
    const double before = d;
    check(!o.reg.get("missing", d) && d == before, "get leaves out untouched");
}

void test_dotted_delegation()
{
    Top t;

    double v = 0.0;
    int n = 0;
    check(t.reg.get("mid.deep.v", v) && v == 3.25, "two-level get");
    check(t.reg.set("mid.deep.v", 7.5), "two-level set");
    check(t.mid.deep.v == 7.5, "two-level set wrote the member");
    check(t.reg.get("mid.n", n) && n == 4, "one-level get");
    check(t.reg.set("mid.n", 9), "one-level set");
    check(t.mid.n == 9, "one-level set wrote the member");

    // heads that are not registry links, and unknown heads
    double d = 0.0;
    check(!t.reg.get("d.mid", d), "dotted through a scalar head");
    check(!t.reg.get("midx.deep.v", d), "unknown dotted head");
    check(!t.reg.get("mid.deep.w", d), "unknown dotted tail");
    check(!t.reg.has("mid.deep.w"), "has unknown dotted name");

    // dotted write through the copy below must not leak here
    Top t2 = t;
    check(t2.reg.set("mid.deep.v", 100.0), "copy: two-level set");
    check(t2.mid.deep.v == 100.0, "copy: set wrote the copy's member");
    check(t.mid.deep.v == 7.5, "copy: original untouched (dotted)");
    check(t2.reg.set("mid.n", 1), "copy: one-level set");
    check(t.mid.n == 9, "copy: original untouched (one-level)");
}

void test_names_and_dump()
{
    Top t;
    t.mid.deep.v = 3.5;
    const std::vector<std::string> names = t.reg.names();
    check(names.size() == 3, "names: 1 own scalar + 2 nested leaves");
    check(names.size() == 3 && names[0] == "d" && names[1] == "mid.n" &&
          names[2] == "mid.deep.v",
          "names: dotted leaf names");

    const char* path = "named_values_dump_test.txt";
    const int tmp = ::open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    check(tmp >= 0, "dump capture file opened");
    if (tmp >= 0)
    {
        std::FILE* f = ::fdopen(tmp, "w");
        check(f != nullptr, "fdopen");
        if (f != nullptr)
        {
            t.reg.dump(f);
            std::fclose(f);
        }
        std::ifstream in(path);
        std::string text;
        std::string line;
        while (std::getline(in, line))
        {
            text += line + "\n";
        }
        check(text.find("d = ") != std::string::npos, "dump own scalar");
        check(text.find("mid.n = 4") != std::string::npos, "dump nested int");
        check(text.find("mid.deep.v = ") != std::string::npos,
              "dump two-level leaf");
        // registry links themselves are not printed as values
        check(text.find("\nmid = ") == std::string::npos,
              "dump: no bare registry line");
    }
    std::remove(path);
}

void test_range_guard()
{
    Outer o;
    double outside = 0.0;
    check(!o.reg.add("outside", outside), "add rejects a foreign scalar");
    check(o.reg.size() == 4, "rejected add did not grow the registry");

    ValueRegistry transient;
    check(!o.reg.add("transient", transient), "add rejects a foreign child");
    check(o.reg.size() == 4, "rejected child add did not grow the registry");
}

void test_copy_move_assignment()
{
    Outer o;
    o.reg.set("x", 11.0);
    o.reg.set("inner.b", 13);

    // copy
    Outer o2 = o;
    check(o2.x == 11.0 && o2.inner.b == 13, "copy carries the values");
    check(o2.reg.set("x", 21.0), "copy: set flat name");
    check(o2.x == 21.0 && o.x == 11.0, "copy: flat set did not touch original");
    check(o2.reg.set("inner.a", 3.5), "copy: set dotted name");
    check(o2.inner.a == 3.5 && o.inner.a == 1.5,
          "copy: dotted set did not touch original");

    // move
    Outer o3 = std::move(o2);
    check(o3.x == 21.0 && o3.inner.b == 13, "move carries the values");
    double xd = 0.0;
    check(o3.reg.get("x", xd) && xd == 21.0, "move: registry still valid");
    check(o3.reg.set("inner.c", false), "move: dotted set after move");
    check(o3.inner.c == false, "move: dotted set after move wrote member");

    // assignment
    Outer o4;
    o4 = o3;
    check(o4.x == 21.0 && o4.inner.b == 13, "assignment carries the values");
    check(o4.reg.get("inner.a", xd) && xd == 3.5,
          "assignment: registry still valid");

    // const access through the registry
    const Outer& oc = o4;
    check(oc.reg.get("x", xd) && xd == 21.0, "const get");
}

void test_double_coercion()
{
    Outer o;

    // get_as_double: each stored kind coerces to double
    double d = -1.0;
    check(o.reg.get_as_double("x", d) && d == 1.0, "double entry as double");
    check(o.reg.get_as_double("y", d) && d == 2.0, "int entry as double");
    check(o.reg.get_as_double("z", d) && d == 0.0, "bool false as 0.0");
    check(o.reg.get_as_double("inner.a", d) && d == 1.5, "dotted double");
    check(o.reg.get_as_double("inner.b", d) && d == 7.0, "dotted int");
    check(o.reg.get_as_double("inner.c", d) && d == 1.0, "dotted bool");

    // set_from_double: values are cast back to the entry's kind
    check(o.reg.set_from_double("y", 9.0), "set int from double");
    check(o.y == 9, "int entry cast back to int");
    check(o.reg.set_from_double("z", 1.0), "set bool true from double");
    check(o.z, "bool entry set true");
    check(o.reg.set_from_double("z", 0.0), "set bool false from double");
    check(!o.z, "bool entry set false");
    check(o.reg.set_from_double("x", 3.25), "set double from double");
    check(o.x == 3.25, "double entry written");

    // truncated int conversion
    check(o.reg.set_from_double("y", 12.9), "set int from fractional double");
    check(o.y == 12, "fractional double truncates to int");

    // dotted delegation through set_from_double / get_as_double
    check(o.reg.set_from_double("inner.b", 21.0), "dotted set");
    check(o.inner.b == 21, "dotted set wrote the member");

    // registry links and unknown names are rejected
    check(!o.reg.get_as_double("inner", d), "link name is not a scalar");
    check(!o.reg.set_from_double("inner", 1.0), "link name cannot be set");
    check(!o.reg.get_as_double("nope", d), "unknown name");
    check(!o.reg.set_from_double("nope", 1.0), "unknown name set");
}

}  // namespace

int main()
{
    test_scalar_access();
    test_double_coercion();
    test_type_mismatch();
    test_dotted_delegation();
    test_names_and_dump();
    test_range_guard();
    test_copy_move_assignment();

    if (failures == 0)
    {
        std::printf("All ValueRegistry checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}
