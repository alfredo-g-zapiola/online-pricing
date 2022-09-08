from typing import Type

import click

from online_pricing.environment import EnvironmentBase
from online_pricing.learner import Learner, TSLearner, UCBLearner
from online_pricing.simulator import Simulator

# TODO(FIL): fix clairvoyant performance
# TODO(FIL): context generation


@click.command()
@click.option(
    "--step", "-s", default=None, help="The step of the simulation, as defined in the project description.", type=int
)
@click.option("--fully-connected", "-fc", is_flag=True, help="Whether to use a fully connected product graph.")
@click.option("--n-days", "-n", default=1000, help="The number of days to simulate.")
@click.option("--learner", "-l", default="TS", help="The learner to use.")
@click.option("--no-plot", "-p", is_flag=True, help="Whether to avoid plotting the results.")
def main(step: int | None, fully_connected: bool, n_days: int, learner: str, no_plot: bool) -> None:

    # TODO: generalize this
    n_arms = 5
    n_prices = 4

    if step is None:
        step = int(input("Step to be run: "))

    # FIXME: qua possiamo anche droppare i default
    match step:
        case 3:
            environment = EnvironmentBase(
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
            )
        case 4:
            environment = EnvironmentBase(
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
            environment = EnvironmentBase(
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

        case _:
            raise ValueError(f"Step {step} does not exists.")

    learner_class: Type[Learner]
    match learner:
        case "TS":
            learner_class = TSLearner
        case "learner_class":
            learner_class = UCBLearner
        case _:
            raise ValueError(f"Learner {learner} does not exists.")

    simulator = Simulator(environment=environment, learner=learner_class)

    try:
        for i in range(n_days):
            simulator.sim_one_day()
    except KeyboardInterrupt:
        print(" !===============! Interrupted !===============! ")

    if not no_plot:
        simulator.plot()


if __name__ == "__main__":
    main()
