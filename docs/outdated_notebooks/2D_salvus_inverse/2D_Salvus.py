# %% [markdown]
# # 2D Acoustic inversion of random perturbations using Optson
# This notebook is intendted for educational and testing purposes.
# It shows how a simple 2D acoustic model can be retrieved using Optson.

# %% [markdown]
# ## 1. Imports, settings and helper functions

# %%
import numpy as np
import pathlib
from itertools import product
from typing import List, Union
from numpy.typing import ArrayLike
from salvus.opt.smoothing import (
    IsotropicModelDependent,
    get_smooth_model,
)
from salvus.mesh.simple_mesh import CartesianHomogeneousAcoustic2D
from salvus.mesh.unstructured_mesh import UnstructuredMesh as UM
import salvus.flow.simple_config as sc
from salvus.flow import api as flow_api
from salvus.flow.collections import EventMisfit
from optson.base_classes.vector import Vector
from salvus.flow.simple_config.receiver.cartesian import Point2D
from salvus.flow.simple_config.source.cartesian import ScalarPoint2D
from salvus.flow.simple_config.stf import FilteredHeaviside
from optson.optimizer import Optimizer
from optson.methods import AdamUpdate
from optson.base_classes.base_problem import StochasticBaseProblem
from optson.base_classes.batch_manager import BasicBatchManager
from optson.base_classes.preconditioner import Preconditioner
from optson.base_classes.base_problem import BaseProblem
from optson.methods import BasicTRUpdate, SteepestDescentUpdate

# Settings
np.random.seed(seed=42)  # Set random state
SITE_NAME = "local"  # only local type runs supported for now.
N_RANKS = 8  # Number of ranks used in Salvus simulations
MAX_FREQ = 1.0 / 10.0
MIN_FREQ = MAX_FREQ * 0.25
PARAMS_TO_INVERT = ["VP"]
VP_INIT = 1500.0
RBO_INIT = 1000.0
ELEMENTS_PER_WAVELENGTH = 2.0
TENSOR_ORDER = 4  # Tensor order for the representation of the model
X_MAX, Y_MAX = 100e3, 100e3
MAX_PERT = 0.05  # Max perturbations from background model
PERT_SL = 0.2  # Perturbation smoothing length  in wavelengths.
END_TIME = np.sqrt(X_MAX**2 + Y_MAX**2) / VP_INIT
WALL_TIME = None


def get_diffusion_model(reference_model_filename, smoothing_length_in_wavelengths):
    imd = IsotropicModelDependent(
        smoothing_length_in_wavelengths=smoothing_length_in_wavelengths,
        reference_frequency_in_hertz=MAX_FREQ,
        reference_velocity="VP",
        reference_model=reference_model_filename,
    )
    return imd.get_diffusion_model(reference_model_filename)


def get_smoothed_model(
    model_to_smooth: Union[pathlib.Path, str],
    diffusion_model: Union[UM, pathlib.Path, str],
    params_to_smooth: List[str] = PARAMS_TO_INVERT,
    tensor_order: int = TENSOR_ORDER,
    verbosity: int = 0,
):
    sims = []
    for param in params_to_smooth:
        sim = sc.simulation.Diffusion(mesh=diffusion_model)
        sim.domain.polynomial_order = tensor_order
        sim.physics.diffusion_equation.courant_number = 0.06
        sim.physics.diffusion_equation.initial_values.filename = str(model_to_smooth)
        sim.physics.diffusion_equation.initial_values.field = f"{param}"
        sim.physics.diffusion_equation.final_values.filename = f"{param}.h5"
        sims.append(sim)
    job = flow_api.run_many_async(
        input_files=sims,
        site_name=SITE_NAME,
        verbosity=verbosity,
        wall_time_in_seconds_per_job=WALL_TIME,
    )
    job.wait(verbosity=verbosity)
    smooth_model = get_smooth_model(job)
    return smooth_model


def mesh_to_vector(m: UM, params_to_invert=PARAMS_TO_INVERT) -> Vector:
    par_list = []
    for param in params_to_invert:
        par_list.append(m.element_nodal_fields[param].flatten())
    return Vector(np.concatenate(par_list))


def vector_to_mesh(x: Vector, target_mesh: UM, params_to_invert=PARAMS_TO_INVERT) -> UM:
    par_vals = np.array_split(x, len(params_to_invert))
    m = target_mesh.copy()

    for idx, param in enumerate(params_to_invert):
        m.element_nodal_fields[param][:] = par_vals[idx].reshape(
            m.element_nodal_fields[param].shape
        )
    return m


# %% [markdown]
# ## 2.1 Define initial model
#

# %%
# Define true and initial model
sm = CartesianHomogeneousAcoustic2D(
    vp=VP_INIT,
    rho=VP_INIT,
    x_max=X_MAX,
    y_max=Y_MAX,
    max_frequency=MAX_FREQ,
    elements_per_wavelength=ELEMENTS_PER_WAVELENGTH,
    tensor_order=TENSOR_ORDER,
)
m_init = sm.create_mesh()
model_dir = pathlib.Path("MODELS")
model_dir.mkdir(parents=True, exist_ok=True)
initial_model_filename = model_dir / "initial_model.h5"
m_init.write_h5(initial_model_filename)
m_init

# %% [markdown]
# ## 2.2 Define true model
# We perturb the background model with smoothed random perturbations.

# %%
# Define random perturbations
m_pert = m_init.copy()
for param in PARAMS_TO_INVERT:
    m_pert.attach_field(param, 100.0 * (np.random.rand(m_pert.npoint) - 0.5))
m_pert.map_nodal_fields_to_element_nodal()
random_perturbation_filename = model_dir / "m_perturbed_raw.h5"
m_pert.write_h5(random_perturbation_filename)

# Create Diffusion model
dm = get_diffusion_model(
    reference_model_filename=initial_model_filename,
    smoothing_length_in_wavelengths=PERT_SL,
)

# Get smooth model perturbations
smooth_pert = get_smoothed_model(
    model_to_smooth=random_perturbation_filename,
    diffusion_model=dm,
)

# Add smoothed perturbations
scaling_factor = (
    VP_INIT * MAX_PERT / np.max(np.abs(smooth_pert.element_nodal_fields["VP"]))
)
m_true = m_init.copy()
for param in PARAMS_TO_INVERT:
    par_init = np.mean(m_init.element_nodal_fields[param])
    scaling_factor = (
        par_init * MAX_PERT / np.max(np.abs(smooth_pert.element_nodal_fields[param]))
    )

    m_true.element_nodal_fields[param][:] += (
        scaling_factor * smooth_pert.element_nodal_fields[param]
    )
m_true.write_h5(model_dir / "true_model.h5")
m_true

# %% [markdown]
# ## 3. Define Dataset

# %%
stf = FilteredHeaviside(
    end_time_in_seconds=END_TIME,
    min_frequency_in_hertz=MIN_FREQ,
    max_frequency_in_hertz=MAX_FREQ,
)

src_x = [0.5]  # 1 source
# srx_y = [0.25, 0.75] # 4 sources
all_srcs = dict()
for i, (x, y) in enumerate(product(src_x, src_x)):
    all_srcs[i] = ScalarPoint2D(
        x=X_MAX * x, y=Y_MAX * y, f=1e1, source_time_function=stf
    )

rec_x = np.linspace(0.15, 0.85, 10)
all_recs = []
for i, (x, y) in enumerate(product(rec_x, rec_x)):
    all_recs.append(
        Point2D(x=X_MAX * x, y=Y_MAX * y, station_code=f"{i}", fields=["phi"])
    )

# %% [markdown]
# ## 4. Create observed data

# %%
absorbing = sc.boundary.Absorbing(
    width_in_meters=0.00 * min(X_MAX, Y_MAX),
    side_sets=["x0", "x1", "y0", "y1"],
    taper_amplitude=0.25 * MIN_FREQ,
)

obs_events = dict()
for src_idx, src in all_srcs.items():
    w = sc.simulation.Waveform(
        mesh=m_true,
        sources=[src],
        receivers=all_recs,
        store_adjoint_checkpoints=False,
    )
    w.physics.wave_equation.boundaries = [absorbing]
    obs_events[src_idx] = flow_api.run(
        input_file=w,
        site_name=SITE_NAME,
        output_folder=f"TRUE_DATA/src_{src_idx}",
        overwrite=True,
        verbosity=0,
        wall_time_in_seconds=WALL_TIME,
        delete_remote_files=True,
    ).get_as_event()


# %% [markdown]
# ## 5. Get gradient for a model and a source, receiver combination


# %%
def get_misfit_and_gradient(m, src, recs, observed_event, src_idx, iteration_name):
    w = sc.simulation.Waveform(
        mesh=m,
        sources=[src],
        receivers=recs,
        store_adjoint_checkpoints=True,
    )
    w.physics.wave_equation.boundaries = [absorbing]
    j_init = flow_api.run(
        input_file=w,
        site_name=SITE_NAME,
        output_folder=f"SYNTHETICS/{iteration_name}/src_{src_idx}",
        overwrite=True,
        wall_time_in_seconds=WALL_TIME,
        delete_remote_files=False,
        verbosity=0,
    )
    ev_init = j_init.get_as_event()
    ev_init.sources = observed_event.sources  # Not sure why this is needed

    event_misfit = EventMisfit(
        observed_event=observed_event,
        synthetic_event=ev_init,
        misfit_function="L2",
        extra_kwargs_misfit_function={},
        receiver_field="phi",
    )
    adjoint_source_filename = pathlib.Path(f"ADJOINT_SOURCES/adj_src_{src_idx}.h5")
    adjoint_source_filename.parent.mkdir(parents=True, exist_ok=True)
    event_misfit.write(filename=adjoint_source_filename)
    w_adjoint = sc.simulation_generator.create_adjoint_waveform_simulation(
        meta_json_forward_run=event_misfit.synthetic_event.meta_json_contents,
        adjoint_source_file=adjoint_source_filename,
        gradient_parameterization="rho-vp",
    )
    output_folder = pathlib.Path(f"GRADIENTS/{iteration_name}/grad_src_{src_idx}")
    flow_api.run(
        input_file=w_adjoint,
        site_name=SITE_NAME,
        output_folder=output_folder,
        overwrite=True,
        wall_time_in_seconds=WALL_TIME,
        verbosity=0,
    )
    j_init.delete(verbosity=0)
    return event_misfit.misfit_value, output_folder / "gradient.h5"


# %% [markdown]
# ## 6. Compute synthetics and show sample gradient (un) comment the below cell if you need this)

# %%
# iteration_info = dict()
# for src_idx, src in all_srcs.items():
#     if src_idx not in iteration_info:
#         iteration_info[src_idx] = dict()
#     observed_event = obs_events[src_idx]
#     (
#         iteration_info[src_idx]["misfit"],
#         iteration_info[src_idx]["gradient"],
#     ) = get_misfit_and_gradient(
#         m=m_init, src=src, recs=all_recs, observed_event=observed_event, src_idx=src_idx, iteration_name="initial"
#     )

# m = None
# for ev in iteration_info.values():
#     grad = ev["gradient"]
#     if m is None:
#         m = UM.from_h5(grad)
#     else:
#         g = UM.from_h5(grad)
#         m.element_nodal_fields["VP"][:] += g.element_nodal_fields["VP"]
# m

# %% [markdown]
# ## Optimization with Optson
# #### Step 1.: Define Problem class

# %%


class InnerProductPreconditioner(Preconditioner):
    def __init__(self, problem):
        self.problem = problem
        self.mm = None

    def get_mm(self):
        if self.mm is None:
            # first iteration
            first_it = list(self.problem.iter_info.keys())[0]
            first_src = list(self.problem.iter_info[first_it].keys())[0]
            m = UM.from_h5(self.problem.iter_info[first_it][first_src]["gradient"])
            fmm = m.element_nodal_fields["FemMassMatrix"]
            valence = m.element_nodal_fields["Valence"]
            mass = fmm / valence / self.problem.misfit_scaling_factor
            self.mm = np.repeat(mass, len(PARAMS_TO_INVERT))
        return self.mm

    def __call__(self, x: Vector) -> Vector:
        return self.get_mm() * x


class Regularizer:
    def __init__(self, diffusion_model=None):
        self.dm = diffusion_model

    def apply_regularization(self, mesh):
        if self.dm is None:
            return mesh
        iv_filename = "MODELS/initial_values.h5"
        mesh.write_h5(iv_filename)
        return get_smoothed_model(iv_filename, dm)


class InverseProblem(BaseProblem):
    """Only for the non-stochastic case."""

    def __init__(self, regularizer: Regularizer = Regularizer(), verbose=False):
        super().__init__()
        self.iter_info = dict()
        self.misfit_scaling_factor = 1e2
        self.verbose = verbose
        self.regularizer = regularizer
        self.preconditioner = InnerProductPreconditioner(self)

    def _compute_misfit_and_gradient(self, x: ArrayLike):
        if x.descriptor in self.iter_info:
            return self.iter_info[x.descriptor]
        self.iter_info[x.descriptor] = dict()

        m = vector_to_mesh(x, m_init)
        m.write_h5(f"MODELS/model_{x.md.iteration}.h5")
        for src_idx, src in all_srcs.items():
            self.iter_info[x.descriptor][src_idx] = dict()
            obs_event = obs_events[src_idx]
            (
                self.iter_info[x.descriptor][src_idx]["misfit"],
                self.iter_info[x.descriptor][src_idx]["gradient"],
            ) = get_misfit_and_gradient(
                m=m,
                src=src,
                recs=all_recs,
                observed_event=obs_event,
                src_idx=src_idx,
                iteration_name=x.descriptor,
            )
        return self.iter_info[x.descriptor]

    def f(self, x: Vector) -> float:
        info = self._compute_misfit_and_gradient(x)
        misfit = 0.0
        for v in info.values():
            misfit += v["misfit"]
        if self.verbose:
            print(misfit, x.descriptor)
        return misfit

    def g(self, x: Vector) -> Vector:
        info = self._compute_misfit_and_gradient(x)
        g = None
        for v in info.values():
            if g is None:
                g = UM.from_h5(v["gradient"])
            else:
                m = UM.from_h5(v["gradient"])
                for param in PARAMS_TO_INVERT:
                    g.element_nodal_fields[param][:] += m.element_nodal_fields[param]
        g = self.regularizer.apply_regularization(g)
        g.write_h5(f"{x.md.iteration}.h5")
        return Vector(mesh_to_vector(g) * self.misfit_scaling_factor)


# %% [markdown]
# ### Step 2: Gradient test the problem

# %%
# from optson.gradient_test import GradientTest

# problem = InverseProblem()
# h = np.logspace(-5, 1, 7)
# gt = GradientTest(x0=mesh_to_vector(m_init), h=h, problem=problem, verbose=True)
# gt.plot()

# %% [markdown]
# ### Step 3. Run optimizer

# %%
# %load_ext autoreload
# %autoreload 2


update = BasicTRUpdate(
    fallback=SteepestDescentUpdate(initial=0.03, step_size_as_percentage=True),
    verbose=True,
)
# !rm mono_batch_cache.h5
dm = get_diffusion_model(
    reference_model_filename=initial_model_filename, smoothing_length_in_wavelengths=0.1
)

problem = InverseProblem(regularizer=Regularizer(diffusion_model=dm), verbose=True)
opt = Optimizer(
    problem=problem,
    store_models=True,
    update=update,
    cache_file="mono_batch_cache.h5",
    verbose=True,
)
opt.stopping_criterion.max_iterations = 5
m_final = opt.iterate(x0=mesh_to_vector(m_init))

# %%
vector_to_mesh(opt.models[-1].x, m_init)

# %%
m_true

# %% [markdown]
#
# ## Stochastic Optimization with Optson

# %%


class StochasticInverseProblem(StochasticBaseProblem):
    """Only for the non-stochastic case."""

    def __init__(
        self,
        batch_size,
        n_samples,
        random_seed=42,
        regularizer: Regularizer = Regularizer(),
        verbose=True,
    ):
        super().__init__(
            batchManager=BasicBatchManager(
                batch_size=batch_size, n_samples=n_samples, random_seed=random_seed
            ),
            preconditioner=InnerProductPreconditioner(self),
        )
        self.regularizer = regularizer
        self.iter_info = dict()
        self.misfit_scaling_factor = 1e6
        self.verbose = verbose

    def _compute_misfit_and_gradient(self, x: ArrayLike, samples):
        # Loop over all required samples and ensure everything is computed.
        if x.descriptor not in self.iter_info:
            self.iter_info[x.descriptor] = dict()
        m = vector_to_mesh(x, m_init)
        for sample in samples:
            src = all_srcs[sample]
            if sample in self.iter_info[x.descriptor]:
                continue
            self.iter_info[x.descriptor][sample] = dict()
            obs_event = obs_events[sample]
            (
                self.iter_info[x.descriptor][sample]["misfit"],
                self.iter_info[x.descriptor][sample]["gradient"],
            ) = get_misfit_and_gradient(
                m=m,
                src=src,
                recs=all_recs,
                observed_event=obs_event,
                src_idx=sample,
                iteration_name=x.descriptor,
            )
        return self.iter_info[x.descriptor]

    def _get_sample_average_gradient(self, x: ArrayLike, samples):
        info = self._compute_misfit_and_gradient(x, samples=samples)
        g = None
        for sample in samples:
            if g is None:
                g = UM.from_h5(info[sample]["gradient"])
            else:
                m = UM.from_h5(info[sample]["gradient"])
                for param in PARAMS_TO_INVERT:
                    g.element_nodal_fields[param][:] += m.element_nodal_fields[param]
        g = self.regularizer.apply_regularization(g)
        return Vector(mesh_to_vector(g) * self.misfit_scaling_factor / len(samples))

    def _get_sample_average_misfit(self, x: ArrayLike, samples):
        info = self._compute_misfit_and_gradient(x, samples=samples)
        misfit = 0.0
        for sample in samples:
            misfit += info[sample]["misfit"]
        misfit /= len(samples)
        if self.verbose:
            print(misfit, x.descriptor)
        return misfit

    def f(self, x: Vector) -> float:
        samples = self.batchManager.get_batch(x.md.iteration)
        return self._get_sample_average_misfit(x, samples)

    def g(self, x: Vector) -> Vector:
        samples = self.batchManager.get_batch(x.md.iteration)
        return self._get_sample_average_gradient(x, samples)

    def f_cg(self, x: Vector) -> float:
        samples = self.batchManager.get_control_group(x.md.iteration)
        return self._get_sample_average_misfit(x, samples)

    def g_cg(self, x: Vector) -> Vector:
        samples = self.batchManager.get_control_group(x.md.iteration)
        return self._get_sample_average_gradient(x, samples)

    def f_cg_previous(self, x: Vector) -> float:
        samples = self.batchManager.get_control_group(x.md.iteration - 1)
        return self._get_sample_average_misfit(x, samples)

    def g_cg_previous(self, x: Vector) -> Vector:
        samples = self.batchManager.get_control_group(x.md.iteration - 1)
        return self._get_sample_average_gradient(x, samples)


# %%
# from optson.gradient_test import GradientTest
# n_samples = len(all_srcs)
# problem = StochasticInverseProblem(
#     batch_size=3, n_samples=n_samples
# )
# h = np.logspace(-4, 2, 7)
# gt = GradientTest(x0=mesh_to_vector(m_init), h=h, problem=problem, verbose=True)
# gt.plot()

# %%


class UpdatePreconditioner(Preconditioner):
    def __call__(self, x: Vector) -> Vector:
        m = vector_to_mesh(x, m_init)
        iv_filename = "MODELS/initial_values.h5"
        m.write_h5(iv_filename)
        return Vector(mesh_to_vector(get_smoothed_model(iv_filename, dm)))


# def smoothing(update):
#     m = vector_to_mesh(update, m_init)
#     iv_filename = "MODELS/initial_values.h5"
#     m.write_h5(iv_filename)
#     return mesh_to_vector(get_smoothed_model(iv_filename, dm))

update = AdamUpdate(
    update_preconditioner=UpdatePreconditioner(),
    epsilon=0.1,
    alpha=15.0,
    relative_epsilon=True,
)
# !rm lbfgs_cache.h5

# update = TrustRegion(fallback=SteepestDescentUpdate(initial=0.03, step_size_as_percentage=True), verbose=True)

n_samples = len(all_srcs)
problem = StochasticInverseProblem(
    batch_size=1, n_samples=n_samples, regularizer=Regularizer(diffusion_model=None)
)

opt = Optimizer(
    problem=problem,
    store_models=True,
    update=update,
    verbose=True,
    cache_file="lbfgs_cache.h5",
)
opt.stopping_criterion.max_iterations = 5
m_final = opt.iterate(x0=mesh_to_vector(m_init))

# %%
vector_to_mesh(opt.models[-1].x, m_init)

# %%
m_true
