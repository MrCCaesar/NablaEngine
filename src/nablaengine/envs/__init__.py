"""Batched, differentiable task environments on the engine: the DeepMind Control Suite's tasks (envs.dmc) and Brax's
(envs.brax), each the original's model, observation, reward, initial-state randomization and control step, for JAX
or PyTorch (envs.env.Task; the models converted for the engine by envs.convert, their kinematics in envs.kin)."""


def tasks():
    """The task names make() takes."""
    from nablaengine.envs import brax as brax_tasks, dmc
    return sorted(dmc.TASKS) + sorted(brax_tasks.TASKS)


def spec(name):
    from nablaengine.envs import brax as brax_tasks, dmc
    table = {**dmc.TASKS, **brax_tasks.TASKS}
    if name not in table:
        raise KeyError(f"unknown task {name!r}; one of {', '.join(sorted(table))}")
    return table[name]()


def make(name, backend="jax", integrator=None, refine=None, iterations=None, reward=None, device=None):
    """envs.env.Task for the task name (tasks()); the arguments as Task's."""
    from nablaengine.envs.env import Task
    return Task(spec(name), backend=backend, integrator=integrator, refine=refine, iterations=iterations,
                reward=reward, device=device)
