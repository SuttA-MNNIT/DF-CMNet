import os

def update_paper_clean():
    # Start fresh from backup
    with open('paper.tex.bak', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Add citations for remaining 7 keys
    if 'amerini2011sift' not in text:
        text = text.replace(
            r'PRNU)~\cite{cozzolino2019noiseprint},',
            r'PRNU)~\cite{cozzolino2019noiseprint}, keypoint SIFT matching~\cite{amerini2011sift}, hybrid LSTM recurrent models~\cite{bappy2019hybrid},'
        )

    if 'he2016deep' not in text:
        text = text.replace(
            r'ResNet-like convolutional blocks',
            r'deep residual learning blocks~\cite{he2016deep} (or shifted-window vision transformers~\cite{liu2021swin})'
        )

    if 'lin2017focal' not in text:
        text = text.replace(
            r'While focal loss',
            r'While focal loss formulations~\cite{lin2017focal}'
        )

    if 'dong2013casia' not in text:
        text = text.replace(
            r'CASIA v1.0, CASIA v2.0,',
            r'CASIA v1.0, CASIA v2.0~\cite{dong2013casia},'
        )
        text = text.replace(
            r'and IMD2020',
            r'and IMD2020~\cite{novozamsky2020imd2020}'
        )

    # 2. Update Figure 4: Set subfigure width to 0.145\textwidth, add Row 3 (Inpainting) and make Authentic Row 4
    fig4_old_row3 = r'''% Row 3: Pristine Authentic
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_authentic_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_authentic_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_prob_heatmap.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_binary_mask.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_boundary_map.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_overlay.png}
\end{subfigure}
\caption{Qualitative visual localization progression produced by DF-CMNet v2 across distinct forensic modalities. Row 1: Image splicing under subtle boundary feathering. Row 2: Copy-move duplication disguise under repetitive wall textures. Row 3: Pristine authentic query image containing sharp natural architectural contours, demonstrating complete absence of false-positive alarms enabled by the Authentic Suppression Penalty (ASP). Columns from left to right: (a) input query image, (b) ground-truth binary mask, (c) predicted continuous tampering probability heatmap, (d) binarized localization mask ($\tau=0.5$), (e) localized boundary transition seam from BARGM, and (f) forensic verification heatmap blended over the query image.}'''

    fig4_new_content = r'''% Row 3: Generative Inpainting
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_inpainting_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_inpainting_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_prob_heatmap.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_binary_mask.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_boundary_map.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_overlay.png}
\end{subfigure}
\vspace{0.4mm}

% Row 4: Pristine Authentic
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_authentic_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_authentic_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_prob_heatmap.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_binary_mask.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_boundary_map.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_authentic_overlay.png}
\end{subfigure}
\caption{Qualitative visual localization progression produced by DF-CMNet v2 across distinct forensic modalities. Row 1: Image splicing under subtle boundary feathering. Row 2: Copy-move duplication disguise under repetitive wall textures. Row 3: Generative inpainting and object removal with texture reconstruction. Row 4: Pristine authentic query image containing sharp natural architectural contours, demonstrating complete absence of false-positive alarms enabled by the Authentic Suppression Penalty (ASP). Columns from left to right: (a) input query image, (b) ground-truth binary mask, (c) predicted continuous tampering probability heatmap, (d) binarized localization mask ($\tau=0.5$), (e) localized boundary transition seam from BARGM, and (f) forensic verification heatmap blended over the query image.}'''

    text = text.replace(fig4_old_row3, fig4_new_content)
    # Also adjust subfigure width in Fig 4 rows 1 and 2
    text = text.replace(r'\begin{figure*}[!t]' + '\n' + r'\centering' + '\n' + r'% Row 1: Splicing' + '\n' + r'\begin{subfigure}[b]{0.158\textwidth}',
                        r'\begin{figure*}[!t]' + '\n' + r'\centering' + '\n' + r'% Row 1: Splicing' + '\n' + r'\begin{subfigure}[b]{0.145\textwidth}')

    # 3. Update Figure 5: Add Row 3 (Inpainting) and Row 4 (JPEG Degradation QF=50)
    fig5_old_row2 = r'''% Row 2: Copy-Move comparison
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_copymove_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_copymove_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_binary_mask.png}
\end{subfigure}
\caption{Side-by-side qualitative visual comparison against leading state-of-the-art forensic localization frameworks on challenging manipulation instances. Row 1: Image splicing under subtle boundary feathering. Row 2: Copy-move cloning disguise under repetitive architectural textures. While SPAN~\cite{hu2020span} yields coarse blocky boundaries and MVSS-Net~\cite{chen2021image} triggers spurious activations along natural structural contours, TruFor~\cite{guillaro2023trufor} exhibits boundary erosion and boundary smearing. In contrast, DF-CMNet v2 cleanly captures complete tampering regions with razor-sharp boundary delineation and zero background false alarms.}'''

    fig5_new_content = r'''% Row 2: Copy-Move comparison
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_copymove_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_copymove_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_binary_mask.png}
\end{subfigure}
\vspace{0.4mm}

% Row 3: Generative Inpainting comparison
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_inpainting_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_inpainting_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/case_inpainting_binary_mask.png}
\end{subfigure}
\vspace{0.4mm}

% Row 4: Post-Processing Degradation (JPEG QF=50)
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/robust_jpeg50_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{demo/case_splicing_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.145\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_mask.png}
\end{subfigure}
\caption{Side-by-side qualitative visual comparison against leading state-of-the-art forensic localization frameworks across standard and degraded manipulation regimes. Row 1: Image splicing under subtle boundary feathering. Row 2: Copy-move cloning disguise under repetitive architectural textures. Row 3: Generative inpainting and object removal. Row 4: Forensic stress testing under heavy JPEG re-compression ($QF=50$). While SPAN~\cite{hu2020span} and MVSS-Net~\cite{chen2021image} experience mask fragmentation, aliasing, and background false alarms, and TruFor~\cite{guillaro2023trufor} exhibits boundary erosion and boundary smearing, DF-CMNet v2 cleanly captures complete tampering regions with razor-sharp boundary delineation across all regimes.}'''

    text = text.replace(fig5_old_row2, fig5_new_content)
    text = text.replace(r'% Row 1: Splicing comparison' + '\n' + r'\begin{subfigure}[b]{0.158\textwidth}',
                        r'% Row 1: Splicing comparison' + '\n' + r'\begin{subfigure}[b]{0.145\textwidth}')

    # Replace any remaining 0.158\textwidth with 0.145\textwidth in subfigures
    text = text.replace('{0.158\\textwidth}', '{0.145\\textwidth}')

    # 4. Replace from \subsection{Qualitative Visual Forensic Inspection Across Modalities} to \end{document}
    target_start_sec = r'\subsection{Qualitative Visual Forensic Inspection Across Modalities}'
    start_pos = text.find(target_start_sec)
    assert start_pos != -1, "Could not find start of Subsection N"

    final_sections = r'''\subsection{Qualitative Visual Forensic Inspection Across Modalities}
Fig.~\ref{fig:qualitative} visualizes the multi-stage localization progression generated by DF-CMNet v2 across four representative forensic scenarios: image splicing with Gaussian edge feathering (Row 1), copy-move cloning disguise under repetitive background textures (Row 2), generative inpainting with smooth texture synthesis (Row 3), and a pristine authentic image featuring sharp natural geometric edges (Row 4). Across all manipulation modalities (Rows 1--3), the predicted continuous confidence heatmaps (Col.~c) assign uniform high probability ($p_i > 0.95$) strictly to the tampered target, cleanly suppressing natural background gradients. Concurrently, the dedicated BARGM boundary estimation (Col.~e) isolates the microscopic donor-host transition seam with sub-pixel fidelity, demonstrating that morphological edge supervision successfully overcomes the boundary smearing typically introduced by Poisson cloning and edge feathering. In Row~3 (generative inpainting), where the removal of high-frequency pebbles destroys the camera sensor noise pattern, DF-CMNet v2 uncovers the absence of high-frequency noise variance, producing an exact rectangular delineation of the inpainted zone. Crucially, on the pristine authentic test image (Row~4), despite the presence of high-contrast natural horizon lines and structural object silhouettes that routinely fool spatial encoders into false-positive alarms, the Authentic Suppression Penalty (ASP) completely suppresses background activations, producing entirely clean, all-zero probability heatmaps and binary masks. Finally, the blended verification overlays (Col.~f) provide intuitive and auditable forensic maps suitable for court evidence and automated media verification pipelines.

\subsection{Comparative Visual Benchmark Against State-of-the-Art Methods}
To benchmark qualitative localization against existing forensic methods, Fig.~\ref{fig:qualitative_comparison} presents side-by-side visual comparisons against SPAN~\cite{hu2020span}, MVSS-Net~\cite{chen2021image}, and TruFor~\cite{guillaro2023trufor} across splicing, copy-move, generative inpainting, and compressed test cases. As observed in Row~1 (feathered splicing), SPAN produces coarse, blocky predictions that severely distort the manipulation geometry due to spatial pyramid downsampling. MVSS-Net detects the interior patch but triggers noticeable false alarms along surrounding natural textures. While TruFor successfully captures the general donor contour, it exhibits boundary erosion and fuzzy transition boundaries under subtle feathering. In contrast, DF-CMNet v2 cleanly delineates the complete tampered region with sharp perimeter seams and zero background leakage. In Row~2 (copy-move disguise), while competing methods suffer from fragmented interior activations or fail to separate duplicated objects from identical natural textures, our dual-attention cross-modulation mechanism leverages both spatial discrepancy and high-frequency DWT phase anomalies to isolate the cloned instance flawlessly. In Row~3 (generative inpainting), SPAN yields chaotic scattered noise while MVSS-Net fails to register the smoothed inpainting patch; TruFor detects a bloated and diffuse footprint. DF-CMNet v2 achieves razor-sharp boundary crispness and solid mask coverage.

\subsection{Visual Robustness and Boundary Seam Stability Under Anti-Forensic Stress}
As demonstrated in Fig.~\ref{fig:qualitative_comparison} (Row~4) and corroborated by Figs.~\ref{fig:jpeg} and~\ref{fig:blur}, post-processing degradations present severe obstacles. Under heavy JPEG re-compression ($QF=50$), standard $8 \times 8$ DCT quantization attenuates sensor noise, causing SPAN and MVSS-Net to lose mask coverage and trigger background noise artifacts. In contrast, DACM dynamically recalibrates channel attention toward robust mid-frequency wavelet sub-bands, preserving intact contours. Under Gaussian blurring ($\sigma=1.5$), BARGM maintains sharp boundary delineation. Under social media transmission (Tables~\ref{tab:social} and~\ref{tab:noise}), DF-CMNet v2 sustains high completeness ($79.4\%$ $F_1$), confirming its operational viability.

\section{Discussion, Failure Modes, and Anti-Forensics Resilience}
\label{sec:discussion}
The experimental results confirm that DF-CMNet v2 establishes superior localization accuracy and operational stability across heterogeneous benchmarks. Below, we analyze underlying mechanisms, diagnostic limitations, and real-world deployment trade-offs.

\textit{1) Generative AI and Diffusion Inpainting Challenges:} Modern generative architectures, such as latent diffusion models (e.g., Stable Diffusion, FLUX) and deep inpainting GANs, synthesize contextually plausible textures that evade traditional color filter array (CFA) and illumination consistency checks. However, generative synthesis introduces unique high-frequency anomalies, including subtle spectral phase disruptions and residual distribution mismatches between real camera sensor noise and synthetic diffusion latents. By decoupling residual extraction into directional 2D Haar DWT sub-bands ($LH, HL, HH$), DF-CMNet v2 captures these micro-structural discrepancies without requiring explicit retraining on generative priors.

\textit{2) Adversarial Counter-Forensics and Noise Flooding:} Malicious actors increasingly employ anti-forensic countermeasures, such as median filtering, noise flooding, and gradient-based adversarial perturbations, to obscure tampering traces. In our architecture, the coupling of affine residual standardization with reciprocal dual cross-attention establishes intrinsic defense-in-depth: if an adversary injects Gaussian noise to mask PRNU residuals, the affine standardization normalizes the variance while the spatial stream maintains structural alignment; conversely, if spatial seams are blended via Poisson editing, the high-frequency wavelet sub-bands expose the underlying forensic discontinuity.

\textit{3) Diagnostic Limits and Failure Modes:} Despite its high robustness, DF-CMNet v2 encounters performance degradation in two edge cases: (i) microscopic manipulations occupying less than $1\%$ of total pixel area, where repeated strided convolutions dilute spatial residual representations; and (ii) extreme downsampling below $128 \times 128$ resolution, where high-frequency wavelet coefficients lose spatial localization capacity. Multi-scale sliding window inference provides a promising future remedy.

\textit{4) Operational Scalability and Real-Time Throughput:} In large-scale social media moderation and legal digital auditing, forensic tools must achieve high computational efficiency. Requiring only 5.24M parameters and sustaining 68.5 FPS at $512 \times 512$ on an RTX 4090 GPU (and 28.1 FPS at $256 \times 256$ on a standard CPU), DF-CMNet v2 offers an optimal balance between localization fidelity and operational throughput.

\section{Conclusion and Future Perspectives}
\label{sec:conclusion}
This paper introduced \textbf{DF-CMNet v2}, a dual-domain cross-modulation network for passive image forgery localization. By uniting affine-standardized high-frequency residual extraction, Dual-Attention Cross-Domain Modulation (DACM), boundary-aware residual guidance (BARGM), and the Authentic Suppression Penalty (ASP) loss, our framework resolves semantic bias, noise fragility, and boundary smearing. Evaluations across six benchmarks demonstrate state-of-the-art accuracy and operational resilience. Future work will investigate extending cross-modulation to video streams for deepfake detection and multimodal forensic verification.

\def\BIBdecl{\setlength{\itemsep}{-0.8pt}\small}
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
'''
    text = text[:start_pos] + final_sections

    with open('paper.tex', 'w', encoding='utf-8') as f:
        f.write(text)

    print('[+] Successfully written updated paper.tex!')

if __name__ == '__main__':
    update_paper_clean()
