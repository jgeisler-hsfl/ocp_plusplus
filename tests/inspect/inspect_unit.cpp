// inspect protocol unit tests (docs/plans/OPTION_INSPECT_PLAN.md, P1):
// typed / double-coerced field access, dotted delegation through a
// nested inspect(), unknown names, leaf-only path semantics, name
// discovery (leaf_names), dump format, the has_inspect trait,
// InspectableLeaf no-op children, private member exposure, and the
// statistics column / table helpers.

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "ocp/utils/inspect.hpp"

namespace
{

using ocp::inspect::get_double;
using ocp::inspect::get_field;
using ocp::inspect::leaf_names;
using ocp::inspect::set_double;
using ocp::inspect::set_field;

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
    template <class V>
    static constexpr void inspect(V&& vis)
    {
        vis("a", &Inner::a);
        vis("b", &Inner::b);
        vis("c", &Inner::c);
    }
};

struct Outer
{
    double x = 1.0;
    int y = 2;
    bool z = false;
    Inner inner;
    template <class V>
    static constexpr void inspect(V&& vis)
    {
        vis("x", &Outer::x);
        vis("y", &Outer::y);
        vis("z", &Outer::z);
        vis("inner", &Outer::inner);
    }
};

// A member that is neither a scalar leaf nor an inspect() child: it
// must stay invisible to the walkers.
struct WithBlob
{
    int x = 1;
    double arr[2] = {1.0, 2.0};
    template <class V>
    static constexpr void inspect(V&& vis)
    {
        vis("x", &WithBlob::x);
        vis("arr", &WithBlob::arr);
    }
};

// A member pointer formed inside the class may be applied from outside
// without re-checking access: private fields can be exposed by name.
struct PrivateBox
{
    template <class V>
    static constexpr void inspect(V&& vis)
    {
        vis("secret", &PrivateBox::secret);
    }
    int peek() const
    {
        return secret;
    }

private:
    int secret = 0;
};

struct LeafBox : ocp::inspect::InspectableLeaf
{
    int x = 5;
};

struct Outer2
{
    double a = 1.0;
    LeafBox box;
    template <class V>
    static constexpr void inspect(V&& vis)
    {
        vis("a", &Outer2::a);
        vis("box", &Outer2::box);
    }
};

struct Row
{
    double r1 = 0.0;
    int r2 = 0;
    bool r3 = false;
    template <class V>
    static constexpr void inspect(V&& vis)
    {
        vis("r1", &Row::r1);
        vis("r2", &Row::r2);
        vis("r3", &Row::r3);
    }
};

struct FakeStats
{
    int iter = 0;
    int status = 0;
    std::vector<Row> rows;
};

void test_scalar_access()
{
    Outer o;
    double xd = 0.0;
    int yi = 0;
    bool zb = true;
    check(get_field(o, "x", xd) && xd == 1.0, "get double leaf");
    check(get_field(o, "y", yi) && yi == 2, "get int leaf");
    check(get_field(o, "z", zb) && zb == false, "get bool leaf");

    check(set_field(o, "x", 3.5), "set double leaf");
    check(o.x == 3.5, "set wrote the member");
    check(set_field(o, "y", 11), "set int leaf");
    check(o.y == 11, "int written");
    check(set_field(o, "z", true), "set bool leaf");
    check(o.z, "bool written");
}

void test_dotted_delegation()
{
    Outer o;
    double da = 0.0;
    int bi = 0;
    check(get_field(o, "inner.a", da) && da == 1.5, "get dotted double");
    check(get_field(o, "inner.b", bi) && bi == 7, "get dotted int");
    check(set_field(o, "inner.a", 4.25), "set dotted double");
    check(o.inner.a == 4.25, "dotted set wrote the member");
    check(set_field(o, "inner.c", false), "set dotted bool");
    check(!o.inner.c, "dotted bool written");
}

// Typed access converts like a plain C++ assignment (compile-checked,
// implicit conversions allowed).
void test_conversions()
{
    Outer o;
    double xd = 0.0;
    check(get_field(o, "y", xd) && xd == 2.0, "int leaf read as double");
    check(set_field(o, "y", 4.9), "double into int leaf compiles");
    check(o.y == 4, "implicit conversion truncates");
}

void test_double_coercion()
{
    Outer o;
    double d = -1.0;
    check(get_double(o, "x", d) && d == 1.0, "double as double");
    check(get_double(o, "y", d) && d == 2.0, "int as double");
    check(get_double(o, "z", d) && d == 0.0, "bool false as 0.0");
    check(get_double(o, "inner.c", d) && d == 1.0, "dotted bool as 1.0");
    check(get_double(o, "inner.b", d) && d == 7.0, "dotted int as double");

    check(set_double(o, "y", 9.0), "set int from double");
    check(o.y == 9, "int entry written");
    check(set_double(o, "z", 1.0), "set bool true from double");
    check(o.z, "bool set true");
    check(set_double(o, "z", 0.0), "set bool false from double");
    check(!o.z, "bool set false");
    check(set_double(o, "y", 12.9), "fractional double into int");
    check(o.y == 12, "truncated to int");
    check(set_double(o, "inner.a", 2.5), "dotted set");
    check(o.inner.a == 2.5, "dotted double written");
}

void test_paths()
{
    Outer o;
    double d = 0.0;
    check(!get_double(o, "nope", d), "unknown name");
    check(!get_double(o, "inner.nope", d), "unknown dotted name");
    check(!get_double(o, "x.a", d), "path through a scalar leaf");
    check(!set_double(o, "nope", 1.0), "unknown name set");
    check(!get_double(o, "inner", d), "child name is not a leaf");
    check(!set_double(o, "inner", 1.0), "child name cannot be set");
    check(!get_double(o, "", d), "empty path");
}

void test_names()
{
    const std::vector<std::string> names = leaf_names<Outer>();
    const std::vector<std::string> want = {"x", "y", "z", "inner.a",
                                           "inner.b", "inner.c"};
    check(names == want, "leaf_names order and content");

    const std::vector<std::string> wb = leaf_names<WithBlob>();
    check(wb.size() == 1 && wb[0] == "x",
          "non-scalar non-inspectable member is invisible");
}

void test_dump()
{
    Outer o;
    FILE* f = std::tmpfile();
    if (f == nullptr)
    {
        check(false, "tmpfile");
        return;
    }
    ocp::inspect::dump(o, f);
    std::fflush(f);
    std::fseek(f, 0, SEEK_END);
    const auto len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string text(static_cast<std::size_t>(len), '\0');
    if (len > 0)
    {
        check(std::fread(text.data(), 1, text.size(), f) == text.size(),
              "fread dump output");
    }
    std::fclose(f);

    check(text.find("x = 1.000000000000e+00\n") != std::string::npos,
          "dump double format");
    check(text.find("y = 2\n") != std::string::npos, "dump int format");
    check(text.find("z = false\n") != std::string::npos,
          "dump bool format");
    check(text.find("inner.a = 1.500000000000e+00\n") != std::string::npos,
          "dotted double dump");
    check(text.find("inner.b = 7\n") != std::string::npos,
          "dotted int dump");
    check(text.find("inner.c = true\n") != std::string::npos,
          "dotted bool dump");
    check(text.find("inner = ") == std::string::npos,
          "children are not printed as values");
}

void test_has_inspect()
{
    using ocp::inspect::has_inspect;
    static_assert(has_inspect<Outer>::value, "Outer has inspect");
    static_assert(has_inspect<Inner>::value, "Inner has inspect");
    static_assert(has_inspect<ocp::inspect::InspectableLeaf>::value,
                  "InspectableLeaf has (no-op) inspect");
    static_assert(has_inspect<LeafBox>::value,
                  "inherited inspect counts");
    static_assert(!has_inspect<double>::value, "double has no inspect");
    static_assert(!has_inspect<int>::value, "int has no inspect");
    check(true, "has_inspect static asserts");
}

void test_inspectable_leaf()
{
    LeafBox lb;
    double d = 0.0;
    check(!get_double(lb, "x", d), "no-op child exposes no leaves");
    check(leaf_names<LeafBox>().empty(), "no-op child has no names");

    Outer2 o2;
    const std::vector<std::string> names = leaf_names<Outer2>();
    check(names.size() == 1 && names[0] == "a",
          "inspectable-leaf child contributes no leaves");
    check(!get_double(o2, "box.x", d), "path into no-op child fails");
}

void test_private_member()
{
    PrivateBox box;
    check(set_double(box, "secret", 7.0), "set private member by name");
    check(box.peek() == 7, "private member written");
    double d = 0.0;
    check(get_double(box, "secret", d) && d == 7.0,
          "get private member by name");
}

void test_stat()
{
    FakeStats st;
    st.iter = 2;
    st.rows = {Row{1.5, 3, true}, Row{2.5, 5, false}};

    const std::vector<std::string> names =
        ocp::inspect::stat_column_names(st);
    const std::vector<std::string> want = {"r1", "r2", "r3"};
    check(names == want, "stat column names");

    std::vector<double> col(2, 0.0);
    check(ocp::inspect::stat_column(st, "r2", col.data()), "known column");
    check(col[0] == 3.0 && col[1] == 5.0, "int column coerced to double");
    check(ocp::inspect::stat_column(st, "r3", col.data()), "bool column");
    check(col[0] == 1.0 && col[1] == 0.0, "bool column coerced");
    check(!ocp::inspect::stat_column(st, "nope", col.data()),
          "unknown column rejected");

    // fill_row: one row's leaves in declaration order
    double fbuf[3] = {0.0, 0.0, 0.0};
    ocp::inspect::fill_row(st.rows[0], fbuf);
    check(fbuf[0] == 1.5 && fbuf[1] == 3.0 && fbuf[2] == 1.0, "fill_row");

    std::vector<double> table(6, 0.0);
    int rows = 0;
    int cols = 0;
    check(ocp::inspect::stat_table(st, table.data(), rows, cols),
          "stat table");
    check(rows == 2 && cols == 3, "table dims");
    check(table[0] == 1.5 && table[1] == 3.0 && table[2] == 1.0,
          "table row 0");
    check(table[3] == 2.5 && table[4] == 5.0 && table[5] == 0.0,
          "table row 1");

    // Zero-row table: names still resolve, extraction is empty.
    FakeStats empty;
    std::vector<double> zcol;
    check(ocp::inspect::stat_column(empty, "r2", zcol.data()),
          "zero rows: known column ok");
    check(!ocp::inspect::stat_column(empty, "nope", zcol.data()),
          "zero rows: unknown column rejected");
    rows = -1;
    cols = -1;
    check(ocp::inspect::stat_table(empty, table.data(), rows, cols),
          "zero-row table");
    check(rows == 0 && cols == 3, "zero-row table dims");
}

void test_copy_semantics()
{
    Outer o;
    set_double(o, "inner.b", 42.0);
    Outer o2 = o;
    int b = 0;
    check(get_field(o2, "inner.b", b) && b == 42,
          "copy carries values through inspect");
}

}  // namespace

int main()
{
    test_scalar_access();
    test_dotted_delegation();
    test_conversions();
    test_double_coercion();
    test_paths();
    test_names();
    test_dump();
    test_has_inspect();
    test_inspectable_leaf();
    test_private_member();
    test_stat();
    test_copy_semantics();

    if (failures == 0)
    {
        std::printf("All inspect checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}
