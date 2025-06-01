from typing import Optional, Union, Callable

import numpy as np
import torch
from cleandiffuser.utils import SUPPORTED_SAMPLING_STEP_SCHEDULE
from cleandiffuser.diffusion.diffusionsde import (
    at_least_ndim, epstheta_to_xtheta, xtheta_to_epstheta,
    ContinuousDiffusionSDE, SUPPORTED_SOLVERS)



class ContinuousDiffusionSDEEX(ContinuousDiffusionSDE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_rg(
            self, xt, t, model, prior,
            forward_level: float = 0.8, n_mc_samples: int = 1):
        
        pred = model["diffusion"](xt, t, None)
        if not self.predict_noise:
            raise NotImplementedError
        else:
            alpha_t, sigma_t = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
            alpha_t = at_least_ndim(alpha_t, xt.dim())
            sigma_t = at_least_ndim(sigma_t, xt.dim())
            x0 = epstheta_to_xtheta(xt, alpha_t, sigma_t, pred)

        x0 = x0 * (1. - self.fix_mask) + prior * self.fix_mask

        rglb_samples = torch.zeros((xt.shape[0], n_mc_samples), device=self.device)
        for i in range(n_mc_samples):
            t_hat = torch.full((x0.shape[0],), 1.0 * forward_level, dtype=torch.float, device=self.device)
            fwd_alpha, fwd_sigma = self.noise_schedule_funcs["forward"](t_hat, **(self.noise_schedule_params or {}))
            fwd_alpha = at_least_ndim(alpha_t, x0.dim())
            fwd_sigma = at_least_ndim(sigma_t, x0.dim())
            xt_hat = x0 * fwd_alpha + fwd_sigma * torch.randn_like(x0)
            xt_hat = xt_hat * (1. - self.fix_mask) + prior * self.fix_mask

            pred = model["diffusion"](xt_hat, t_hat, None)
            if not self.predict_noise:
                raise NotImplementedError
            else:
                x0_hat = epstheta_to_xtheta(xt_hat, fwd_alpha, fwd_sigma, pred)

            x0_hat = x0_hat * (1. - self.fix_mask) + prior * self.fix_mask

            diff = x0 - x0_hat.detach()
            rglb_sample = diff.reshape(diff.shape[0], -1).norm(p=2.0, dim=1)

            rglb_samples[:, i] = rglb_sample.view(-1)

        rglb = rglb_samples.mean(dim=-1)
        return rglb

    def low_density_guidance(
            self, xt, t, alpha, sigma, model, w,
            forward_level, n_mc_samples, prior, pred):
        
        if w == 0.0:
            return pred
        else:
            with torch.enable_grad():
                xt = xt.detach().requires_grad_(True)
                rg = self.compute_rg(xt, t, model, prior, 
                    forward_level=forward_level, n_mc_samples=n_mc_samples)
                grad = torch.autograd.grad(rg.sum(), xt)[0]

            if self.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

            return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False,
            # ----------- Low Density Guidance Params ------------ #
            w_ldg: float = 0.0,
            rg_forward_level: float = 0.8,
            n_mc_samples: int = 1,
            prior: torch.Tensor = None,
        ):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        pred = self.low_density_guidance(
            xt, t, alpha, sigma, model, w_ldg, rg_forward_level, 
            n_mc_samples, prior, pred)

        return pred, logp

    # ==================== Sampling: Solving SDE/ODE ======================

    def sample(
            self,
            # ---------- the known fixed portion ---------- #
            prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            solver: str = "ddpm",
            n_samples: int = 1,
            sample_steps: int = 5,
            sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            use_ema: bool = True,
            temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg=None,
            mask_cfg=None,
            w_cfg: float = 0.0,
            condition_cg=None,
            w_cg: float = 0.0,
            # ----------- Diffusion-X sampling ----------
            diffusion_x_sampling_steps: int = 0,
            # ----------- Warm-Starting -----------
            warm_start_reference: Optional[torch.Tensor] = None,
            warm_start_forward_level: float = 0.3,
            # ----------- Low-Density Guidance -----------
            w_ldg: float = 0.0,
            rg_forward_level: float = 0.8,
            n_mc_samples: int = 5,
            # ------------------ others ------------------ #
            requires_grad: bool = False,
            preserve_history: bool = False,
            **kwargs,
    ):
        """Sampling.
        
        Inputs:
        - prior: torch.Tensor
            The known fixed portion of the input data. Should be in the shape of generated data.
            Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.
        
        - solver: str
            The solver for the reverse process. Check `supported_solvers` property for available solvers.
        - n_samples: int
            The number of samples to generate.
        - sample_steps: int
            The number of sampling steps. Should be greater than 1.
        - sample_step_schedule: Union[str, Callable]
            The schedule for the sampling steps.
        - use_ema: bool
            Whether to use the exponential moving average model.
        - temperature: float
            The temperature for sampling.
        
        - condition_cfg: Optional
            Condition for Classifier-free-guidance.
        - mask_cfg: Optional
            Mask for Classifier-guidance.
        - w_cfg: float
            Weight for Classifier-free-guidance.
        - condition_cg: Optional
            Condition for Classifier-guidance.
        - w_cg: float
            Weight for Classifier-guidance.
            
        - diffusion_x_sampling_steps: int
            The number of diffusion steps for diffusion-x sampling.
        
        - warm_start_reference: Optional[torch.Tensor]
            Reference data for warm-starting sampling. `None` indicates no warm-starting.
        - warm_start_forward_level: float
            The forward noise level to perturb the reference data. Should be in the range of `[0., 1.]`, where `1` indicates pure noise.
        
        - requires_grad: bool
            Whether to preserve gradients.
        - preserve_history: bool
            Whether to preserve the sampling history.
            
        Outputs:
        - x0: torch.Tensor
            Generated samples. Be in the shape of `(n_samples, *x_shape)`.
        - log: dict
            The log dictionary.
        """
        assert solver in SUPPORTED_SOLVERS, f"Solver {solver} is not supported."

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((n_samples, sample_steps + 1, *prior.shape)) if preserve_history else None, }

        model = self.model if not use_ema else self.model_ema

        prior = prior.to(self.device)
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            warm_start_forward_level = self.epsilon + warm_start_forward_level * (1. - self.epsilon)
            fwd_alpha, fwd_sigma = self.noise_schedule_funcs["forward"](
                torch.ones((1,), device=self.device) * warm_start_forward_level, **(self.noise_schedule_params or {}))
            """modification: start from mean"""
            xt = warm_start_reference * fwd_alpha# + fwd_sigma * torch.randn_like(warm_start_reference)
        else:
            xt = torch.randn_like(prior) * temperature
        xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(requires_grad):
            condition_vec_cfg = model["condition"](condition_cfg, mask_cfg) if condition_cfg is not None else None
            condition_vec_cg = condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(warm_start_reference, torch.Tensor) and warm_start_forward_level > 0.:
            t_diffusion = [self.t_diffusion[0], warm_start_forward_level]
        else:
            t_diffusion = self.t_diffusion

        if isinstance(sample_step_schedule, str):
            if sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[sample_step_schedule](
                    t_diffusion, sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {sample_step_schedule} is not supported.")
        elif callable(sample_step_schedule):
            sample_step_schedule = sample_step_schedule(t_diffusion, sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            sample_step_schedule, **(self.noise_schedule_params or {}))
        
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((sample_steps + 1,), device=self.device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * diffusion_x_sampling_steps + list(range(1, sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((n_samples,), sample_step_schedule[i], dtype=torch.float32, device=self.device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, w_cfg, condition_vec_cg, w_cg, requires_grad,
                w_ldg, rg_forward_level, n_mc_samples, prior)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif solver == "sde_dpmsolver_1":
                xt = ((alphas[i - 1] / alphas[i]) * xt -
                      2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta +
                      sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * D +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))
                else:
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if preserve_history:
                log["sample_history"][:, sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.classifier is not None and w_cg != 0.:
            with torch.no_grad():
                t = torch.zeros((n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        # compute_rg (restoration gap)
        log["rg"] = self.compute_rg(
            xt, t, model, prior, rg_forward_level, n_mc_samples)

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        return xt, log
