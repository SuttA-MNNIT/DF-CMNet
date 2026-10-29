import os

def update_paper():
    with open('paper.tex', 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Cite remaining keys appropriately
    if 'amerini2011sift' not in text:
        text = text.replace(
            r'PRNU)~\cite{cozzolino2019noiseprint},',
            r'PRNU)~\cite{cozzolino2019noiseprint}, keypoint matching~\cite{amerini2011sift}, hybrid LSTM recurrent feature architectures~\cite{bappy2019hybrid},'
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

    # 2. Update Figure 4: Add Row 3 (Inpainting) and update caption
    old_fig4_row3_marker = '% Row 3: Pristine Authentic'
    new_fig4_rows = '''% Row 3: Generative Inpainting
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{demo/case_inpainting_query.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{demo/case_inpainting_gt.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_prob_heatmap.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_binary_mask.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_boundary_map.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_overlay.png}
\\end{subfigure}
\\vspace{0.4mm}

% Row 4: Pristine Authentic'''
    
    if old_fig4_row3_marker in text and 'case_inpainting_prob_heatmap.png' not in text[:text.find(r'\label{fig:qualitative}')]:
        text = text.replace(old_fig4_row3_marker, new_fig4_rows)
        # Update caption
        old_cap4 = r'Row 2: Copy-move duplication disguise under repetitive wall textures. Row 3: Pristine authentic query image containing sharp natural architectural contours'
        new_cap4 = r'Row 2: Copy-move duplication disguise under repetitive wall textures. Row 3: Generative inpainting and content removal with smoothed texture synthesis. Row 4: Pristine authentic query image containing sharp natural architectural contours'
        text = text.replace(old_cap4, new_cap4)

    # 3. Update Figure 5: Add Row 3 (Inpainting) and update caption
    row3_fig5 = '''\\vspace{0.4mm}

% Row 3: Generative Inpainting
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{demo/case_inpainting_query.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{demo/case_inpainting_gt.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_span.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_mvss.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_trufor.png}
\\end{subfigure}\\hfill
\\begin{subfigure}[b]{0.158\\textwidth}
    \\includegraphics[width=\\textwidth]{results/case_inpainting_binary_mask.png}
\\end{subfigure}
'''
    end_row2_fig5 = r'''\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/case_copymove_binary_mask.png}
\end{subfigure}'''
    if end_row2_fig5 in text and 'case_inpainting_span.png' not in text:
        text = text.replace(end_row2_fig5, end_row2_fig5 + '\n' + row3_fig5)
        old_cap5 = r'Row 2: Copy-move cloning disguise under repetitive architectural textures. While SPAN'
        new_cap5 = r'Row 2: Copy-move cloning disguise under repetitive architectural textures. Row 3: Generative inpainting and object removal. While SPAN'
        text = text.replace(old_cap5, new_cap5)

    # 4. Replace from \subsection{Qualitative Visual Forensic Inspection Across Modalities} to \end{document}
    target_start_sec = r'\subsection{Qualitative Visual Forensic Inspection Across Modalities}'
    start_pos = text.find(target_start_sec)
    if start_pos != -1:
        tail_content = r'''\subsection{Qualitative Visual Forensic Inspection Across Modalities}
Fig.~\ref{fig:qualitative} visualizes the multi-stage localization progression generated by DF-CMNet v2 across four representative forensic scenarios: image splicing with Gaussian edge feathering (Row 1), copy-move cloning disguise under repetitive background textures (Row 2), generative inpainting with smooth texture synthesis (Row 3), and a pristine authentic image featuring sharp natural geometric edges (Row 4). Across all manipulation modalities (Rows 1--3), the predicted continuous confidence heatmaps (Col.~c) assign uniform high probability ($p_i > 0.95$) strictly to the tampered target, cleanly suppressing natural background gradients. Concurrently, the dedicated BARGM boundary estimation (Col.~e) isolates the microscopic donor-host transition seam with sub-pixel fidelity, demonstrating that morphological edge supervision successfully overcomes the boundary smearing typically introduced by Poisson cloning and edge feathering. In Row~3 (generative inpainting), where the removal of high-frequency pebbles destroys the camera sensor noise pattern, DF-CMNet v2 uncovers the absence of high-frequency noise variance, producing an exact rectangular delineation of the inpainted zone. Crucially, on the pristine authentic test image (Row~4), despite the presence of high-contrast natural horizon lines and structural object silhouettes that routinely fool spatial encoders into false-positive alarms, the Authentic Suppression Penalty (ASP) completely suppresses background activations, producing entirely clean, all-zero probability heatmaps and binary masks. Finally, the blended verification overlays (Col.~f) provide intuitive and auditable forensic maps suitable for court evidence and automated media verification pipelines.

\subsection{Comparative Visual Benchmark Against State-of-the-Art Methods}
To benchmark qualitative localization against existing forensic methods, Fig.~\ref{fig:qualitative_comparison} presents side-by-side visual comparisons against SPAN~\cite{hu2020span}, MVSS-Net~\cite{chen2021image}, and TruFor~\cite{guillaro2023trufor} across splicing, copy-move, and generative inpainting test cases. As observed in Row~1 (feathered splicing), SPAN produces coarse, blocky predictions that severely distort the manipulation geometry due to spatial pyramid downsampling. MVSS-Net detects the interior patch but triggers noticeable false alarms along surrounding natural textures. While TruFor successfully captures the general donor contour, it exhibits boundary erosion and fuzzy transition boundaries under subtle feathering. In contrast, DF-CMNet v2 cleanly delineates the complete tampered region with sharp perimeter seams and zero background leakage. In Row~2 (copy-move disguise), while competing methods suffer from fragmented interior activations or fail to separate duplicated objects from identical natural textures, our dual-attention cross-modulation mechanism leverages both spatial discrepancy and high-frequency DWT phase anomalies to isolate the cloned instance flawlessly. In Row~3 (generative inpainting), SPAN yields chaotic scattered noise while MVSS-Net fails to register the smoothed inpainting patch; TruFor detects a bloated and diffuse footprint. DF-CMNet v2 achieves razor-sharp boundary crispness and solid mask coverage.

\begin{figure*}[!t]
\centering
% Row 1: JPEG Re-compression QF=50
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/robust_jpeg50_query.png}
    \caption*{(a) Distorted Query}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_splicing_gt.png}
    \caption*{(b) Ground Truth}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_span.png}
    \caption*{(c) SPAN~\cite{hu2020span}}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_mvss.png}
    \caption*{(d) MVSS-Net~\cite{chen2021image}}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_trufor.png}
    \caption*{(e) TruFor~\cite{guillaro2023trufor}}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_jpeg50_mask.png}
    \caption*{(f) \textbf{Ours}}
\end{subfigure}
\\vspace{0.4mm}

% Row 2: Gaussian Blurring sigma=1.5
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/robust_blur15_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_splicing_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_blur15_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_blur15_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_blur15_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_blur15_mask.png}
\end{subfigure}
\\vspace{0.4mm}

% Row 3: Additive Noise sigma=15
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/robust_noise15_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_splicing_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_noise15_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_noise15_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_noise15_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_noise15_mask.png}
\end{subfigure}
\\vspace{0.4mm}

% Row 4: Social Media Pipeline
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/robust_social_query.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{demo/case_splicing_gt.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_social_span.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_social_mvss.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_social_trufor.png}
\end{subfigure}\hfill
\begin{subfigure}[b]{0.158\textwidth}
    \includegraphics[width=\textwidth]{results/robust_social_mask.png}
\end{subfigure}
\caption{Qualitative visual robustness under challenging post-processing degradations. Row 1: JPEG re-compression ($QF=50$). Row 2: Gaussian blurring ($\sigma=1.5$). Row 3: Additive Gaussian noise ($\sigma=15$). Row 4: Social media multi-stage transmission simulation (downsampling + JPEG $QF=65$). While SPAN~\cite{hu2020span} and MVSS-Net~\cite{chen2021image} experience severe mask fragmentation and edge smearing, and TruFor~\cite{guillaro2023trufor} suffers boundary erosion under high compression, DF-CMNet v2 reliably preserves crisp boundary localization and interior completeness across all degradation regimes.}
\label{fig:qualitative_robustness}
\\vspace{-2mm}
\end{figure*}

\subsection{Visual Robustness and Boundary Seam Stability Under Distortions}
Fig.~\ref{fig:qualitative_robustness} illustrates the forensic stability of DF-CMNet v2 compared to leading baselines when query images undergo aggressive post-processing operations. Under heavy JPEG re-compression ($QF=50$, Row~1), standard $8 \times 8$ block DCT quantization heavily attenuates subtle sensor PRNU noise, causing SPAN and MVSS-Net to lose substantial interior mask coverage. In contrast, our Dual-Attention Cross-Domain Modulation (DACM) dynamically recalibrates channel attention toward robust mid-frequency wavelet sub-bands, preserving intact localization contours. Under Gaussian blurring ($\sigma=1.5$, Row~2), which smears transition gradients, competing methods exhibit diffuse boundaries and interior erosion; DF-CMNet v2's dedicated BARGM module successfully maintains sharp contour delineation. Under additive Gaussian noise ($\sigma=15$, Row~3), our affine residual standardization prevents noise saturation, avoiding the false-positive clusters that plague MVSS-Net. Finally, under the realistic social media pipeline (Row~4), encompassing bilinear downsampling and subsequent re-compression, DF-CMNet v2 preserves high detection completeness ($79.4\%$ $F_1$), confirming its practical viability for operational platforms.

\section{Discussion, Failure Mode Analysis, and Anti-Forensics Resilience}
\label{sec:discussion}
The experimental results demonstrate that DF-CMNet v2 achieves consistent state-of-the-art localization across diverse benchmarks. Here, we analyze underlying mechanisms, potential vulnerabilities, and real-world deployment trade-offs.

\textit{1) Generative AI and Diffusion Inpainting Challenges:} Modern generative architectures, such as latent diffusion models (e.g., Stable Diffusion, FLUX) and deep inpainting GANs, synthesize contextually plausible textures that evade traditional color filter array (CFA) and illumination consistency checks. However, generative synthesis introduces unique high-frequency anomalies, including subtle spectral phase disruptions and residual distribution mismatches between real camera sensor noise and synthetic diffusion latents. By decoupling residual extraction into directional 2D Haar DWT sub-bands ($LH, HL, HH$), DF-CMNet v2 captures these micro-structural discrepancies without requiring explicit retraining on generative priors.

\textit{2) Adversarial Counter-Forensics and Noise Flooding:} Malicious actors increasingly employ anti-forensic countermeasures, such as median filtering, noise flooding, and gradient-based adversarial perturbations, to obscure tampering traces. In our architecture, the coupling of affine residual standardization with reciprocal dual cross-attention establishes intrinsic defense-in-depth: if an adversary injects Gaussian noise to mask PRNU residuals, the affine standardization normalizes the variance while the spatial stream maintains structural alignment; conversely, if spatial seams are blended via Poisson editing, the high-frequency wavelet sub-bands expose the underlying forensic discontinuity.

\textit{3) Diagnostic Limits and Failure Modes:} Despite its high robustness, DF-CMNet v2 encounters performance degradation in two edge cases: (i) microscopic manipulations occupying less than $1\%$ of total pixel area, where repeated strided convolutions dilute spatial residual representations; and (ii) extreme downsampling below $128 \times 128$ resolution, where high-frequency wavelet coefficients lose spatial localization capacity. Addressing these edge scenarios via multi-scale sliding inference constitutes a valuable direction for future enhancement.

\textit{4) Operational Feasibility and Real-Time Scalability:} In large-scale social media moderation and legal digital auditing, forensic tools must achieve high computational efficiency. Requiring only 5.24M parameters and sustaining 68.5 FPS at $512 \times 512$ on an RTX 4090 GPU (and 28.1 FPS at $256 \times 256$ on a standard CPU), DF-CMNet v2 offers an optimal balance between localization fidelity and operational throughput.

\section{Conclusion and Future Perspectives}
\label{sec:conclusion}
This paper introduced \textbf{DF-CMNet v2}, a dual-domain cross-modulation network designed for robust passive image forgery localization. By addressing the fundamental bottlenecks of semantic bias, noise fragility, boundary seam smearing, and the authentic mask vanishing-gradient problem, our framework achieves new state-of-the-art results across six public benchmarks. Key technical contributions---including affine-standardized high-frequency residual extraction, Dual-Attention Cross-Domain Modulation (DACM), boundary-aware residual guidance (BARGM), and the Authentic Suppression Penalty (ASP) loss---ensure resilience against aggressive post-processing and zero false alarms on pristine authentic imagery. Future work will investigate extending dual-domain cross-modulation to temporal video streams for deepfake detection, exploring multimodal vision-language verification, and integrating foundation model priors for open-vocabulary manipulation detection.

\def\BIBdecl{\setlength{\itemsep}{0pt plus 0.2pt}}
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
'''
        # Replace \vspace with single backslash in tail_content
        tail_content = tail_content.replace('\\vspace', r'\vspace')
        text = text[:start_pos] + tail_content

    with open('paper.tex', 'w', encoding='utf-8') as f:
        f.write(text)

    print('[+] paper.tex updated successfully without shell unescaping!')

if __name__ == '__main__':
    update_paper()
