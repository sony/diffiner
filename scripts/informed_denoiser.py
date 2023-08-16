import torch
import tqdm

from guided_diffusion.diffiner_util import create_model_and_diffusion


def get_informed_denoiser(diffusion):

    """
    get informed denoiser, which denoise data with given noise map.
        args:
            diffusion : instance of diffusion class.
            currently "diffusion" must be the instance of GaussianDiffusion in guided_diffusion/gaussian_diffusion.py.
            use_ddim (deprecated): flag whether DDIM sampling is used or not when sampling.
            eta_ddim (deprecated): The parameter for DDIM sampling.
        retunrs:
            informed_denoiser : function
    """

    def informed_denoiser(
        model,
        noisy_data,
        noise_map,
        clip_denoised=False,
        model_kwargs=None,
        etaA_ddrm=1.0,
        etaB_ddrm=1.0,
    ):

        """
        conduct the informed denoising with given noise map.
        args:
            model : score model, whose output should contain "pred_xstart" field. That it the estimation of x_0 given noisy input x_t.
            noisy_data : (bsz, c, h, w) noisy data to be denoised.
            noise_map  : (bsz, c, h, w) the amplitude of Gaussian noise at each pixel.
            clip_denoised (bool) : if True, clip the denoised signal into [-1, 1].
            model_kwargs (dict) : if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
            etaA_ddrm (double) : Hyper parameter when sampling in (sigma_t < noise_map)
                if etaA_ddrm is near to 0.0, the method uses more information from the observation.
                if etaA_ddrm is near to 1.0, the method uses more information from the generative model.
            etaB_ddrm (double) : Hyper parameter when sampling in sigma_t > noise_map
                if etaB_ddrm is near to 0.0, the method uses more information from the generative model.
                if etaB_ddrm is near to 1.0, the method uses more information from the observation.
        """

        device = next(model.parameters()).device

        etaA_ddrm = torch.tensor(etaA_ddrm, device=device).float()
        etaB_ddrm = torch.tensor(etaB_ddrm, device=device).float()

        x_t = torch.randn_like(noisy_data)

        indices = list(range(diffusion.num_timesteps))[::-1]
        b, c, h, w = noisy_data.shape

        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * b, device=device)
            with torch.no_grad():

                # x_t = \sqrt(cumalpha_t) * x_0 + \sqrt(1.0 - cumalpha_t) * z
                sqrt_1_m_cumalpha = torch.sqrt(
                    torch.tensor(1.0 - diffusion.alphas_cumprod_prev[i], device=device)
                ).float()
                sqrt_cumalpha = torch.sqrt(
                    torch.tensor(
                        diffusion.alphas_cumprod_prev[i], device=device
                    ).float()
                )
                mask_sigmat_is_larger = 1.0 * (
                    sqrt_1_m_cumalpha[None, None, None, None]
                    > sqrt_cumalpha * noise_map
                )

                scale_x0_at_t = sqrt_cumalpha
                scaled_noisy_data = noisy_data * scale_x0_at_t

                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0

                # Estimation of x_0 with diffusion model
                res_p_mean_variance = diffusion.p_mean_variance(
                    model,
                    x_t,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )

                est_x_0 = res_p_mean_variance["pred_xstart"] * scale_x0_at_t

                # For sigma_t > noise_map
                sigma_for_larger_sigmat = torch.sqrt(
                    mask_sigmat_is_larger
                    * (
                        sqrt_1_m_cumalpha[None, None, None, None] ** 2
                        - (etaB_ddrm ** 2) * ((sqrt_cumalpha ** 2) * (noise_map ** 2))
                    )
                )
                data_for_larger_sigmat = (
                    (1.0 - etaB_ddrm) * est_x_0
                    + etaB_ddrm * scaled_noisy_data
                    + nonzero_mask * sigma_for_larger_sigmat * torch.randn_like(x_t)
                )

                # For sigma_t < noise_map
                sigma_for_smaller_sigmat = torch.sqrt(
                    (1.0 - mask_sigmat_is_larger)
                    * (sqrt_1_m_cumalpha ** 2)
                    * (etaA_ddrm ** 2)
                )
                coef = (sqrt_1_m_cumalpha / sqrt_cumalpha) / (noise_map + 1e-5)
                data_for_smaller_sigmat = (
                    est_x_0
                    + torch.sqrt(1 - etaA_ddrm ** 2)
                    * coef
                    * (scaled_noisy_data - est_x_0)
                ) + nonzero_mask * sigma_for_smaller_sigmat * torch.randn_like(x_t)

                x_t = (
                    data_for_smaller_sigmat * (1.0 - mask_sigmat_is_larger)
                    + data_for_larger_sigmat * mask_sigmat_is_larger
                )

        return x_t

    return informed_denoiser


def get_improved_informed_denoiser(diffusion):

    """
    get informed denoiser, which denoise data with given noise map.
        args:
            diffusion : instance of diffusion class.
            currently "diffusion" must be the instance of GaussianDiffusion in guided_diffusion/gaussian_diffusion.py.
        retunrs:
            informed_denoiser : function
    """

    def informed_denoiser_v2(
        model,
        noisy_data,
        noise_map,
        clip_denoised=False,
        model_kwargs=None,
        etaA=1.0,
        etaB=1.0,
        etaC=0.0,
        inp_mask=None,
        etaD=1.0,
    ):

        """
        conduct the informed denoising with given noise map.
        args:
            model : score model, whose output should contain "pred_xstart" field. That it the estimation of x_0 given noisy input x_t.
            noisy_data : (bsz, c, h, w) noisy data to be denoised.
            noise_map  : (bsz, c, h, w) the amplitude of Gaussian noise at each pixel.
            clip_denoised (bool) : if True, clip the denoised signal into [-1, 1].
            model_kwargs (dict) : if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
            etaA (double) : Hyper parameter when sampling in (sigma_t < noise_map)
            etaB (double) : Hyper parameter when sampling in sigma_t > noise_map
            etaC (double) : Hyper parameter when sampling in sigma_t < noise_map
            inp_mask : (bsz, c, h, w) : if inp_mask == 1.0 with an element, the element is generated with DDIM sampling (with parameter etaD).
        """

        assert (etaA ** 2 + etaC ** 2) <= 1.0 + 1e-10, "etaA^2 + etaC^2 <= 1.0"

        device = next(model.parameters()).device

        etaA = torch.tensor(etaA, device=device).float()
        etaB = torch.tensor(etaB, device=device).float()
        etaC = torch.tensor(etaC, device=device).float()

        x_t = torch.randn_like(noisy_data)
        if inp_mask is None:
            inp_mask = torch.zeros_like(noisy_data)

        indices = list(range(diffusion.num_timesteps))[::-1]
        b, c, h, w = noisy_data.shape

        for i in tqdm.tqdm(indices):
            t = torch.tensor([i] * b, device=device)
            with torch.no_grad():

                # x_t = \sqrt(cumalpha_t) * x_0 + \sqrt(1.0 - cumalpha_t) * z
                sqrt_1_m_cumalpha = torch.sqrt(
                    torch.tensor(1.0 - diffusion.alphas_cumprod_prev[i], device=device)
                ).float()
                sqrt_cumalpha = torch.sqrt(
                    torch.tensor(
                        diffusion.alphas_cumprod_prev[i], device=device
                    ).float()
                )
                mask_sigmat_is_larger = 1.0 * (
                    sqrt_1_m_cumalpha[None, None, None, None]
                    > sqrt_cumalpha * noise_map
                )

                scale_x0_at_t = sqrt_cumalpha
                scaled_noisy_data = noisy_data * scale_x0_at_t

                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0

                # Estimation of x_0 with diffusion model
                res_p_mean_variance = diffusion.p_mean_variance(
                    model,
                    x_t,
                    t,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )

                est_x_0 = res_p_mean_variance["pred_xstart"] * scale_x0_at_t

                # For sigma_t > noise_map
                sigma_for_larger_sigmat = torch.sqrt(
                    mask_sigmat_is_larger
                    * (
                        sqrt_1_m_cumalpha[None, None, None, None] ** 2
                        - (etaB ** 2) * ((sqrt_cumalpha ** 2) * (noise_map ** 2))
                    )
                )
                data_for_larger_sigmat = (
                    (1.0 - etaB) * est_x_0
                    + etaB * scaled_noisy_data
                    + nonzero_mask * sigma_for_larger_sigmat * torch.randn_like(x_t)
                )

                # For sigma_t < noise_map
                eps = diffusion._predict_eps_from_xstart(
                    x_t, t, res_p_mean_variance["pred_xstart"]
                )  # eps is not scaled (original scale)
                data_for_smaller_sigmat_A = eps * sqrt_1_m_cumalpha

                coef = (sqrt_1_m_cumalpha / sqrt_cumalpha) / (noise_map + 1e-5)
                data_for_smaller_sigmat_C = coef * (scaled_noisy_data - est_x_0)

                masked_v_a = (
                    data_for_smaller_sigmat_A * (1.0 - mask_sigmat_is_larger) + 1e-5
                )
                masked_v_c = (
                    data_for_smaller_sigmat_C * (1.0 - mask_sigmat_is_larger) + 1e-5
                )
                cos_sim = torch.nn.CosineSimilarity(dim=0)(
                    torch.flatten(masked_v_a), torch.flatten(masked_v_c)
                )

                # assert ((etaA**2 + etaC**2 + 2*etaA*etaC*cos_sim) <= 1.0 + 1e-10), "etaA^2 + etaC^2 + 2*etaA*etaC*cos_sim <= 1.0"
                if (etaA ** 2 + etaC ** 2 + 2 * etaA * etaC * cos_sim) > 1.0 + 1e-10:
                    x_t = None
                    break
                sigma_for_smaller_sigmat = torch.sqrt(
                    (1.0 - mask_sigmat_is_larger)
                    * (sqrt_1_m_cumalpha ** 2)
                    * (1.0 - etaA ** 2 - etaC ** 2 - 2 * etaA * etaC * cos_sim)
                )

                data_for_smaller_sigmat = (
                    est_x_0
                    + etaA * data_for_smaller_sigmat_A
                    + etaC * data_for_smaller_sigmat_C
                    + nonzero_mask * sigma_for_smaller_sigmat * torch.randn_like(x_t)
                )

                sigma_for_inp_mask = torch.sqrt(
                    (1.0 - mask_sigmat_is_larger)
                    * (sqrt_1_m_cumalpha ** 2)
                    * (1.0 - etaD)
                )
                data_for_inp_mask = (
                    est_x_0
                    + etaD * data_for_smaller_sigmat_A
                    + nonzero_mask * sigma_for_inp_mask * torch.randn_like(x_t)
                )

                data_for_smaller_sigmat = (
                    1.0 - inp_mask
                ) * data_for_smaller_sigmat + inp_mask * data_for_inp_mask

                x_t = (
                    data_for_smaller_sigmat * (1.0 - mask_sigmat_is_larger)
                    + data_for_larger_sigmat * mask_sigmat_is_larger
                )

        return x_t

    return informed_denoiser_v2
