# Architecture

## 1. Purpose and Scope

This project is a C++17 framework for developing and evaluating numerical algorithms for optimal control problems (OCPs).

The primary goal is **not** to implement a single OCP solver, but to provide a stable mathematical problem interface that can be used by many different optimization algorithms and research implementations.

The architecture therefore separates:

1. the mathematical problem definition and
2. the solver algorithm (different solver algorithms shall consume the same problem definition).

---

# 2. Design Principles

The architecture follows these principles:

* **Static polymorphism:** the numerical core shall primarily use C++17 templates rather than virtual dispatch. The exact C++ mechanism may use traits, compile-time detection, CRTP, C++17 SFINAE, or a combination thereof. The final mechanism should be chosen for clarity and compile-time guarantees rather than abstraction for its own sake.
* **Compile-time structure:** intrinsic problem dimensions must be available at compile time to allow for static allocation of all problem und solution variables and optimized matrix operations.
* **Compact problem definition:** the optimization problem, constisting of the model dynamics, cost function (stage and terminal), constraints (stage and terminal) and their derivatives, and the dimensions of all relevant vectors is encapsuled in a class (possibly using sub-classes or structure, e.g. for dimensions, dynamics, cost and constraints) that adheres to a defined "template-interface". Concrete implementations of a specific Problem are strictly user implemented.
* **Discrete dynamics first:** at first only discrete dynamic systems are considered. It is up to the user to implement discretisation schemes and derivatives.
* **Efficient linear algebra** using the eigen3 library. All vectors and matrices are fixed-size eigen3 obejcts.
* **Runtime flexibility:** quantities such as the horizon may be runtime-variable.
* **Representation independence:** mathematical objects such as trajectories should not be defined by their underlying storage format.
* **Minimal abstraction:** abstractions should represent meaningful mathematical or architectural concepts rather than anticipated future requirements.
* **Separation of parameters:** parameters of the model and problem are managed by the user supplied class. Parameters and options of the solver algorithm are managed individually by each solver implementation, sharing only a generaly high-level interface that allows to define arbitry option variables and access them as member variables of a solver-specific options class or by a string name via a map.
* **Separation of internal variables:** each solver implementation stores its own internal states and intermediate variables (like e.g. multipliers, barrier parameters, regularization parameters, or solver-internal slack variables). Only solution variables common to all solvers (especially the trajectories of states and inputs) are accessible via a shared interface.

---

# 3. Scalar Type

Numerical types should preferably be template parameters where practical.

Conceptually:

```cpp
template<class Scalar>
class Problem;
```

This allows the framework to support different scalar types without making `double` an intrinsic requirement of the mathematical model.

---

# 4. Static and Runtime Dimensions and Data Types

The architecture distinguishes between **structural dimensions** and **instance dimensions**.

Structural dimensions such as `nx`, `nu`, and constraint dimensions normally describe the mathematical type and therefore belong at compile time.

The horizon is different. The same mathematical problem type may naturally be used with different horizon lengths:

```text
N = 50
N = 100
N = 200
```

Therefore the architecture shall support a runtime horizon.

At the same time, compile-time horizons should remain possible for applications where they are beneficial.

The architecture must therefore accommodate both:

```text
fixed structural dimensions + fixed horizon
```

and

```text
fixed structural dimensions + runtime horizon
```

without making either implementation mandatory at this stage.

---

# 5. Horizon Semantics and Trajectory Representation

The temporal extent of a quantity must be explicit.

For a standard discrete-time OCP,

$$
x_0,\ldots,x_N
$$

and

$$
u_0,\ldots,u_{N-1}
$$

have different extents.

Typical quantities therefore have:

```text
state trajectory       N + 1
control trajectory     N
dynamics residual      N
path constraints       N
terminal constraints   1
```

The framework shall not rely on an implicit convention that every horizon-dependent object has the same length.

The framework shall provide a semantic representation for horizon-dependent quantities.

The mathematical interface should conceptually expose objects such as:

```cpp
Trajectory<State, N>
Trajectory<Control, N>
```

rather than defining trajectories directly as Eigen matrices.

A trajectory represents a sequence of stage-wise quantities. Its physical storage is an implementation detail.

Stage-wise access should be natural, for example:

```cpp
x[k]
u[k]
constraint[k]
slack[k]
```

or through an equivalent interface.

---

# 6. Objective

The architecture must allow alternative formulations such as:

* exact Hessians,
* Gauss-Newton approximations,
* other valid approximations,
* or absence of second-order information for algorithms that do not require it.

The derivative source is not part of the mathematical interface.

---

# 7. Constraints

The problem interface shall represent constraints independently of their numerical treatment.

Supported categories shall include, where applicable:

* Box constraints
* Linear in states and inputs constraints
* General equality constraints
* General inequality constraints
* Individually defined slacks (soft-constraints) on all types of constraints

The interface must remain sufficiently general to accommodate additional formulations without requiring changes to the solver architecture.
