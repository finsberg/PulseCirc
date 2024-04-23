from pathlib import Path
import matplotlib.pyplot as plt
from structlog import get_logger
import csv

logger = get_logger()

class DataCollector:
    def __init__(self, outdir: Path | None = None) -> None:
        self.times = []
        self.activations = []
        self.volumes = []
        self.pressures = []
        self.flows = []
        self.ao_pressures = []
        self.outdir = outdir

    def collect(
        self,
        time: float,
        activation: float,
        volume: float,
        pressure: float,
        flow: float,
        p_ao: float,
    ) -> None:
        logger.info(
            "Collecting data",
            time=time,
            activation=activation,
            volume=volume,
            pressure=pressure,
            flow=flow,
            p_ao=p_ao,
        )
        self.times.append(time)
        self.activations.append(activation)
        self.volumes.append(volume)
        self.pressures.append(pressure)
        self.flows.append(flow)
        self.ao_pressures.append(p_ao)
        if self.outdir is not None:
            self.save(time)

    def save(self, t: float) -> None:
        if self.outdir is None:
            return
        fig, axs = plt.subplots(
            2, 2, figsize=(10, 10)
        )  # Create a figure and two subplots
        axs[0, 0].plot(self.times, self.activations)
        axs[0, 0].set_ylabel("Activation (kPa)")
        axs[0, 0].set_xlabel("Time (ms)")
        axs[0, 1].plot(self.volumes, self.pressures)
        axs[0, 1].set_ylabel("Pressure (kPa)")
        axs[0, 1].set_xlabel("Volume (ml)")
        axs[1, 0].plot(self.times, self.flows)
        axs[1, 0].set_ylabel("Outflow (ml/s)")
        axs[1, 0].set_xlabel("Time (ms)")
        axs[1, 1].plot(self.times, self.pressures, label="LV Pressure")
        axs[1, 1].plot(self.times, self.ao_pressures, label="Aortic Pressure")
        axs[1, 1].legend()
        axs[1, 1].set_ylabel("Pressure (kPa)")
        axs[1, 1].set_xlabel("Time (ms)")
        fig.savefig(Path(self.outdir) / f"results.png")
        plt.close(fig)
        with open(Path(self.outdir) / "results_data.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Time [ms]",
                    "Activation [kPa]",
                    "Volume [ml]",
                    "LV Pressure [kPa]",
                    "Aortic Pressure [kPa]",
                    "Outflow[ml/ms]",
                ]
            )
            for time, activation, vol, pres_val, ao_pres_val, flow in zip(
                self.times,
                self.activations,
                self.volumes,
                self.pressures,
                self.ao_pressures,
                self.flows,
            ):
                writer.writerow([time, activation, vol, pres_val, ao_pres_val, flow])
