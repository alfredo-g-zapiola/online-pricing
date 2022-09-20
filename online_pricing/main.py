import warnings
from typing import Any

import click
from tqdm import tqdm

from online_pricing.common.environment import EnvironmentBase
from online_pricing.common.learner import LearnerFactory
from online_pricing.common.simulator import Simulator
from online_pricing.helpers.tracer import Tracer

warnings.filterwarnings("ignore")


@click.command()
@click.option(
    "--step", "-s", default=None, help="The step o  f the simulation, as defined in the project description.", type=int
)
@click.option("--fully-connected", "-fc", is_flag=True, help="Whether to use a fully connected product graph.")
@click.option("--n-days", "-n", default=100, help="The number of days to simulate.")
@click.option("--no-plot", "-p", is_flag=True, help="Whether to avoid plotting the results.")
@click.option("--n-sims", "-ns", default=30, help="How many simulations we carry out")
@click.option("--sliding-window", "-sw", default=False, help="In step 6, whether to use UCB learners with sliding window")
def main(
    step: int | None,
    fully_connected: bool,
    n_days: int,
    no_plot: bool,
    n_sims: int,
    sliding_window: bool,
) -> None:
    print()
    print()
    print(" !==============================! Simulation Starting !==============================! ")
    print()
    print()

    if step is None:
        step = int(input("Step to be run: "))

    match step:
        case 3:
            environments: list[EnvironmentBase] = [
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters={
                        "fully_connected": fully_connected,
                        "learner_class": "TS",
                        "context_generation": False,
                        "uncertain_alpha": False,
                        "group_unknown": True,
                        "lambda": 0.5,
                        "uncertain_demand_curve": False,
                        "unknown_demand_curve": False,
                        "uncertain_quantity_bought": False,
                        "unknown_quantity_bought": False,
                        "uncertain_product_weights": False,
                        "unknown_product_weights": False,
                    },
                ),
            ]

        case 4:
            environments = [
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters={
                        "fully_connected": fully_connected,
                        "learner_class": "TS",
                        "context_generation": False,
                        "uncertain_alpha": True,
                        "group_unknown": True,
                        "lambda": 0.5,
                        "uncertain_demand_curve": False,
                        "unknown_demand_curve": True,
                        "uncertain_quantity_bought": True,
                        "unknown_quantity_bought": True,
                        "uncertain_product_weights": False,
                        "unknown_product_weights": False,
                    },
                ),
            ]
        case 5:
            environments = [
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters={
                        "fully_connected": fully_connected,
                        "learner_class": "TS",
                        "context_generation": False,
                        "uncertain_alpha": False,
                        "group_unknown": True,
                        "lambda": 0.5,
                        "uncertain_demand_curve": False,
                        "unknown_demand_curve": False,
                        "uncertain_quantity_bought": False,
                        "unknown_quantity_bought": False,
                        "uncertain_product_weights": True,
                        "unknown_product_weights": True,
                    },
                ),
            ]
        case 6:
            base_parameters = {
                "fully_connected": fully_connected,
                "learner_class": "TS",
                "context_generation": False,
                "uncertain_alpha": False,
                "group_unknown": True,
                "lambda": 0.5,
                "uncertain_demand_curve": False,
                "unknown_demand_curve": False,
                "uncertain_quantity_bought": False,
                "unknown_quantity_bought": False,
                "uncertain_product_weights": False,
                "unknown_product_weights": False,
                "shifting_demand_curve": True,
                "Ucb_sliding_window": sliding_window,
            }

            environments = [
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters=base_parameters | {"learner_class": "MUCB"},
                ),
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters=base_parameters | {"learner_class": "SWUCB"},
                ),
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters=base_parameters | {"learner_class": "MUCB"},
                ),
            ]
        case 7:
            environments = [
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters={
                        "fully_connected": fully_connected,
                        "context_generation": True,
                        "learner_class": "CGUCB",
                        "uncertain_alpha": False,
                        "group_unknown": True,
                        "lambda": 0.5,
                        "uncertain_demand_curve": False,
                        "unknown_demand_curve": False,
                        "uncertain_quantity_bought": False,
                        "unknown_quantity_bought": False,
                        "uncertain_product_weights": False,
                        "unknown_product_weights": False,
                        "shifting_demand_curve": False,
                    },
                ),
            ]

        case _:
            raise ValueError(f"Step {step} does not exists.")

    try:
        for environment in environments:
            learner_args: dict[str, Any] = {
                "window_size": 10,
                "w": 10,
                "beta": 3,
                "gamma": 0.5,
                "context_window": 14,
                "n_features": 2,
            }
            learner_factory = LearnerFactory(environment.learner_class, **learner_args)

            tracer = Tracer(n_sims, n_days)

            run_simulator(n_sims, n_days, environment, tracer, learner_factory)

            if not no_plot:
                tracer.plot_day()
                tracer.plot_total()

    except KeyboardInterrupt:
        print()
        print()
        print(" !==============================! Interrupted !==============================! ")
        return

    finally:
        print()
        print()
        print(" !==============================! End of Simulation !==============================! ")
        print()
        print()


def run_simulator(
    n_samples: int, n_days: int, environment: EnvironmentBase, tracer: Tracer, learner_factory: LearnerFactory
) -> None:
    for n in range(n_samples):
        simulator = Simulator(environment, int(n * 4314), tracer, learner_factory)
        for _ in tqdm(range(n_days), desc=f"Simulating realization {n + 1}"):
            simulator.sim_one_day()
        tracer.add_daily_data(rewards=simulator.reward_tracer.avg_reward, regrets=simulator.reward_tracer.regret, sample=n)

        if n != n_samples - 1:
            tracer.new_day()


if __name__ == "__main__":
    main()
