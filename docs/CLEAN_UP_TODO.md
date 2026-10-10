- remove all references to local paths from all project files
- ~~in examples/pendulum_on_cart/generated/PendulumGen.hp I can see that the acados adapter uses a finite difference for the gradient. that was not intended. of course, the adapter shall use all generated casadi code, especially for derivatives and especially the fused versions shall be used in the spirit of the framework, they shall be "handed-through" to the discretized problem formulation even if that means that two or three separate functions are necessary, each calling the integrator in a different way. same same goes, of course, for nonlinear costs and constraints.~~ (done — see `docs/plans/GENERATED_DERIVATIVES_PLAN.md`)
- refactor to use implicit ode formulation for implicit integrators.
- add an explicit switch to use GAUSS_NEWTON or no dynamics hessian at all, if none is implemented.
- add linear soft constraint penalties, just like in acados/hpipm
- rename include/ocp/solvers/acados/sqp.hpp to acados.hpp and the main class to AcadosSolver because there might follow other sqp based solvers. check if all currently public methods need to be public.
- rework all code documentation to be more precise, informative and not too long (doxygen style), name inputs and outputs. leave the genesis (reference to the implementation phases) and the acados original code only as supplementary info.

