from typing import Type

import click

from online_pricing.environment import EnvironmentBase
from online_pricing.learner import Learner, TSLearner, UCBLearner
from online_pricing.simulator import Simulator
from online_pricing.tracer import Tracer


@click.command()
@click.option(
    "--step", "-s", default=None, help="The step o  f the simulation, as defined in the project description.", type=int
)
@click.option("--fully-connected", "-fc", is_flag=True, help="Whether to use a fully connected product graph.")
@click.option("--n-days", "-n", default=100, help="The number of days to simulate.")
@click.option("--learner", "-l", default="TS", help="The learner to use.")
@click.option("--no-plot", "-p", is_flag=True, help="Whether to avoid plotting the results.")
@click.option("--n-sims", "-ns", default=30, help="How many simulations we carry out")
def main(step: int | None, fully_connected: bool, n_days: int, learner: str, no_plot: bool, n_sims: int) -> None:

    learner_class: Type[Learner]
    match learner:
        case "TS":
            learner_class = TSLearner
        case "learner_class":
            learner_class = UCBLearner
        case _:
            raise ValueError(f"Learner {learner} does not exists.")

    if step is None:
        step = int(input("Step to be run: "))

    match step:
        case 3:
            environments = (
                EnvironmentBase(
                    n_products=5,
                    n_groups=3,
                    hyperparameters={
                        "fully_connected": fully_connected,
                        "context_generation": False,
                        "uncertain_alpha": False,
                        "group_unknown": True,
                        "lambda": 0.5,
                        "uncertain_demand_curve": False,
                        "uncertain_quantity_bought": True,
                        "uncertain_product_weights": False,
                    },
                ),
            )

        case 4:
            environments = EnvironmentBase(
                n_products=5,
                n_groups=3,
                hyperparameters={
                    "fully_connected": fully_connected,
                    "context_generation": False,
                    "uncertain_alpha": True,
                    "group_unknown": True,
                    "lambda": 0.5,
                    "uncertain_demand_curve": False,
                    "uncertain_quantity_bought": False,
                    "uncertain_product_weights": False,
                },
            )
        case 5:
            environments = EnvironmentBase(
                n_products=5,
                n_groups=3,
                hyperparameters={
                    "fully_connected": fully_connected,
                    "context_generation": False,
                    "uncertain_alpha": False,
                    "group_unknown": True,
                    "lambda": 0.5,
                    "uncertain_demand_curve": False,
                    "uncertain_quantity_bought": False,
                    "uncertain_product_weights": True,
                },
            )
        case 7:
            environments = EnvironmentBase(
                n_products=5,
                n_groups=3,
                hyperparameters={
                    "fully_connected": fully_connected,
                    "context_generation": False,
                    "uncertain_alpha": False,
                    "group_unknown": True,
                    "lambda": 0.5,
                    "uncertain_demand_curve": False,
                    "uncertain_quantity_bought": False,
                    "uncertain_product_weights": True,
                    "shifting_demand_curve": True,
                },
            )

        case _:
            raise ValueError(f"Step {step} does not exists.")

    for environment in environments:
        tracer = Tracer(n_sims, n_days)
        run_simulator(n_sims, n_days, environment, tracer=tracer)
        if not no_plot:

            tracer.plot_total()


def run_simulator(n_samples: int, n_days: int, environment: EnvironmentBase, tracer: Tracer) -> None:
    try:
        for n in range(n_samples):
            simulator = Simulator(environment=environment, seed=int(n * 4314), tracer=tracer)
            for i in range(n_days):
                simulator.sim_one_day()
            tracer.add_daily_data(rewards=simulator.reward_tracer.avg_reward, sample=n)

    except KeyboardInterrupt:
        print(" !===============! Interrupted !===============! ")


if __name__ == "__main__":
    main()
