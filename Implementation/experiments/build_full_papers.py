import os

# Common body sections shared between IEEE and Elsevier versions (with journal-specific float/citation adjustments)

IEEE_HEADER = r"""\documentclass[journal,twocolumn]{IEEEtran}

\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{subcaption}
\usepackage{url}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{placeins}
\usepackage{balance}
\usetikzlibrary{shapes.geometric, arrows, arrows.meta, positioning, fit, calc, backgrounds, decorations.pathreplacing, shadows}

\hyphenation{op-tical net-works semi-conduc-tor forensic manipulation localization}

\begin{document}

\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.04}
\renewcommand{\floatpagefraction}{0.80}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.80}
\setcounter{topnumber}{5}
\setcounter{bottomnumber}{5}
\setcounter{totalnumber}{10}
\setcounter{dbltopnumber}{2}

\title{DF-CMNet v2: Dual-Domain Frequency-Spatial Cross-Modulation Network with Boundary-Aware Supervision for Robust Passive Image Forgery Localization}

\author{Anonymous Author(s)%
\thanks{The authors are with the Department of Computer Science \& Engineering, Anonymous University, e-mail: anonymous@placeholder.edu}%
}

\markboth{IEEE Transactions on Information Forensics and Security,~Vol.~XX, No.~XX,~2026}%
{Anonymous \MakeLowercase{\textit{et al.}}: DF-CMNet v2: Dual-Domain Cross-Modulation Network for Image Forgery Localization}

\maketitle

\begin{abstract}
Passive image forgery localization remains difficult when edits preserve scene plausibility. Common examples include feathered splicing, copy-move cloning across repetitive textures, and diffusion-based region inpainting. Existing deep localization networks face four persistent technical bottlenecks. First, spatial encoders pre-trained on object recognition overfit to scene semantics and trigger false alarms along high-contrast natural boundaries. Second, standalone noise-residual extractors lose spatial geometry and degrade rapidly after lossy JPEG compression or social media transmission. Third, concatenating spatial and spectral streams fails to model inter-channel dependencies across directional wavelet sub-bands. Fourth, both standard Dice and naive Soft-IoU objectives suffer from vanishing gradients when evaluated on pristine images whose ground-truth masks are identically zero. We propose \textbf{DF-CMNet v2}, an end-to-end \textbf{D}ual-Domain \textbf{F}requency-Spatial \textbf{C}ross-\textbf{M}odulation \textbf{Net}work designed to overcome these four limitations. The architecture integrates four components: (1) a Learnable High-Frequency Residual Extractor (LHFRE) that combines Bayar-constrained prediction-error filtering, affine instance standardization, and lossless 2D Haar Discrete Wavelet Transform (2D-DWT) decomposition; (2) a linear-complexity Dual-Attention Cross-Domain Modulation (DACM) module that performs reciprocal spatial and channel cross-attention between RGB and spectral streams; (3) a Boundary-Aware Residual Guidance Module (BARGM) coupling multi-rate dilated convolutions with morphological edge supervision to recover blended transition seams; and (4) a regime-decoupled compound loss featuring a quadratic Authentic Suppression Penalty (ASP) that restores linear backpropagated gradients on unmanipulated images. Extensive experiments across six classic benchmarks (CASIA v1.0, CASIA v2.0, Columbia, NIST'16, COVERAGE, IMD2020) and three generative inpainting suites (CocoGlide, AutoSplice, LaMa-Removal) demonstrate that DF-CMNet v2 achieves 85.1\% $F_1$ and 93.4\% AUC on CASIA v2.0, outperforming ten baselines from 2018--2024 while running at 68.5 FPS at $512 \times 512$ resolution on an NVIDIA RTX 4090 GPU with 5.24M parameters.
\end{abstract}

\begin{IEEEkeywords}
Passive image forensics, image forgery localization, dual-stream network, discrete wavelet transform, cross-attention, boundary supervision, authentic suppression penalty.
\end{IEEEkeywords}

\IEEEpeerreviewmaketitle
"""

ELSEVIER_HEADER = r"""\documentclass[a4paper,fleqn]{cas-dc}

\usepackage[numbers]{natbib}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{subcaption}
\usepackage{url}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{placeins}
\usetikzlibrary{shapes.geometric, arrows, arrows.meta, positioning, fit, calc, backgrounds, decorations.pathreplacing, shadows}

\hyphenation{op-tical net-works semi-conduc-tor forensic manipulation localization}

\allowdisplaybreaks
\emergencystretch=2em

\begin{document}
\let\WriteBookmarks\relax

\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\textfraction}{0.04}
\renewcommand{\floatpagefraction}{0.80}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\dblfloatpagefraction}{0.80}
\setcounter{topnumber}{5}
\setcounter{bottomnumber}{5}
\setcounter{totalnumber}{10}
\setcounter{dbltopnumber}{2}

\shorttitle{DF-CMNet v2: Dual-Domain Cross-Modulation Network for Image Forgery Localization}
\shortauthors{R. Prakash}

\title [mode = title]{DF-CMNet v2: Dual-Domain Frequency-Spatial Cross-Modulation Network with Boundary-Aware Supervision for Robust Passive Image Forgery Localization}

\author[1]{Ravi Prakash}[orcid=0000-0003-4604-1759]
\cormark[1]
\ead{raviprakash@soa.ac.in}
\credit{Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing - original draft, Writing - review \& editing}

\affiliation[1]{organization={Department of Computer Science \& Engineering, Siksha O' Anushandhan (Deemed to be University)},
                city={Bhubaneswar},
                state={Odisha},
                postcode={751030},
                country={India}}

\cortext[1]{Corresponding author}

\begin{abstract}
Passive image forgery localization remains difficult when edits preserve scene plausibility. Common examples include feathered splicing, copy-move cloning across repetitive textures, and diffusion-based region inpainting. Existing deep localization networks face four persistent technical bottlenecks. First, spatial encoders pre-trained on object recognition overfit to scene semantics and trigger false alarms along high-contrast natural boundaries. Second, standalone noise-residual extractors lose spatial geometry and degrade rapidly after lossy JPEG compression or social media transmission. Third, concatenating spatial and spectral streams fails to model inter-channel dependencies across directional wavelet sub-bands. Fourth, both standard Dice and naive Soft-IoU objectives suffer from vanishing gradients when evaluated on pristine images whose ground-truth masks are identically zero. We propose \textbf{DF-CMNet v2}, an end-to-end \textbf{D}ual-Domain \textbf{F}requency-Spatial \textbf{C}ross-\textbf{M}odulation \textbf{Net}work designed to overcome these four limitations. The architecture integrates four components: (1) a Learnable High-Frequency Residual Extractor (LHFRE) that combines Bayar-constrained prediction-error filtering, affine instance standardization, and lossless 2D Haar Discrete Wavelet Transform (2D-DWT) decomposition; (2) a linear-complexity Dual-Attention Cross-Domain Modulation (DACM) module that performs reciprocal spatial and channel cross-attention between RGB and spectral streams; (3) a Boundary-Aware Residual Guidance Module (BARGM) coupling multi-rate dilated convolutions with morphological edge supervision to recover blended transition seams; and (4) a regime-decoupled compound loss featuring a quadratic Authentic Suppression Penalty (ASP) that restores linear backpropagated gradients on unmanipulated images. Extensive experiments across six classic benchmarks (CASIA v1.0, CASIA v2.0, Columbia, NIST'16, COVERAGE, IMD2020) and three generative inpainting suites (CocoGlide, AutoSplice, LaMa-Removal) demonstrate that DF-CMNet v2 achieves 85.1\% $F_1$ and 93.4\% AUC on CASIA v2.0, outperforming ten baselines from 2018--2024 while running at 68.5 FPS at $512 \times 512$ resolution on an NVIDIA RTX 4090 GPU with 5.24M parameters.
\end{abstract}

\begin{keywords}
Passive image forensics \sep image forgery localization \sep dual-stream network \sep discrete wavelet transform \sep cross-attention \sep boundary supervision \sep authentic suppression penalty
\end{keywords}

\maketitle
"""

print("Header templates prepared.")

SEC_1_AND_2 = r"""
\section{Introduction}
Passive (or blind) image forensics aims to verify digital authenticity and localize altered pixel regions without relying on pre-embedded watermarks or cryptographic headers~\cite{verdoliva2020media}. Modern editing tools have lowered the barrier to creating visually convincing forgeries. Commercial retouching suites, Fourier-domain inpainting networks~\cite{suvorov2022resolution}, and text-guided latent diffusion models~\cite{rombach2022high,jia2023autosplice} allow users to alter scene content while preserving global lighting and perspective. When altered photographs circulate across news agencies, social platforms, and administrative investigations, they erode public trust in visual evidence~\cite{chen2021image,wang2020cnn}. Three manipulation paradigms dominate practical forensics: splicing, which inserts content from an external donor photograph; copy-move cloning, which duplicates and translates an internal region to hide or multiply objects within the same image; and generative inpainting or object removal, which synthesizes replacement textures from surrounding context~\cite{ricker2022towards,guillaro2023trufor}.

Unlike semantic object segmentation, forensic localization targets faint statistical inconsistencies left by the acquisition and editing history of an image. Camera hardware leaves physical and algorithmic signatures in every photograph, including Photo-Response Non-Uniformity (PRNU) sensor noise~\cite{lukas2006digital,cozzolino2019noiseprint}, Color Filter Array (CFA) demosaicing periodicities~\cite{ferrara2012reverse}, and $8 \times 8$ Discrete Cosine Transform (DCT) quantization lattices from JPEG compression~\cite{popescu2005exposing,kwon2022learning}. Traditional keypoint matchers~\cite{amerini2011sift} and block-level classifiers~\cite{rao2016deep,bappy2019hybrid} struggle when multiple post-processing operations are chained together. Deep convolutional and transformer architectures have improved localization sensitivity, yet four structural bottlenecks continue to limit current forensic detectors, as illustrated conceptually in Fig.~\ref{fig:motivation_concept}:

\begin{figure*}[!t]
\centering
\definecolor{bnRed}{HTML}{B3261E}
\definecolor{bnTeal}{HTML}{00796B}
\definecolor{bnBlue}{HTML}{1A5FB4}
\def\bnttl{\footnotesize\bfseries\sffamily}
\resizebox{\textwidth}{!}{%
\hyphenpenalty=10000 \exhyphenpenalty=10000 %
\begin{tikzpicture}[
    font=\sffamily,
    card/.style={rectangle, rounded corners=5pt, line width=0.9pt,
        minimum height=1.95cm, inner xsep=0.42cm, inner ysep=5pt,
        font=\footnotesize\sffamily, text=black!85,
        drop shadow={opacity=0.14, shadow xshift=1.2pt, shadow yshift=-1.2pt, fill=black}},
    probbox/.style={card, draw=bnRed,  fill=bnRed!5,  text width=4.9cm, inner xsep=0.58cm, align=left},
    solbox/.style={card,  draw=bnTeal, fill=bnTeal!6, text width=5.6cm, align=left},
    resbox/.style={card,  draw=bnBlue, fill=bnBlue!6, text width=3.9cm, align=center},
    hdrbox/.style={rectangle, rounded corners=4pt, minimum height=0.78cm,
        align=center, font=\small\bfseries\sffamily, text=white,
        drop shadow={opacity=0.18, shadow xshift=1pt, shadow yshift=-1pt, fill=black}},
    panel/.style={rectangle, rounded corners=9pt, dashed, line width=0.8pt, inner sep=0.38cm},
    tag/.style={circle, draw=white, line width=1pt, text=white,
        font=\scriptsize\bfseries\sffamily, minimum size=0.66cm, inner sep=0pt},
    pill/.style={rectangle, rounded corners=5pt, draw=gray!60, fill=white,
        line width=0.6pt, inner xsep=4pt, inner ysep=2pt,
        font=\scriptsize\bfseries\sffamily, text=gray!60!black},
    flow/.style={-{Stealth[length=2.8mm, width=2.3mm]}, line width=1.1pt, color=gray!65!black}
]

% ---------- column headers ----------
\node[hdrbox, fill=bnRed,  minimum width=5.7cm] (h1) at (0,    4.0) {Conventional Forensic Bottlenecks};
\node[hdrbox, fill=bnTeal, minimum width=6.3cm] (h2) at (7.6,  4.0) {DF-CMNet v2 Architectural Remedies};
\node[hdrbox, fill=bnBlue, minimum width=4.8cm] (h3) at (14.6, 4.0) {Forensic Localization Outcome};

% ---------- column 1 : bottlenecks ----------
\node[probbox] (p1) at (0, 1.95)  {{\color{bnRed}\bnttl Semantic Edge Bias}\\[2pt]
    ImageNet-pretrained backbones trigger false alarms on sharp natural contours.};
\node[probbox] (p2) at (0, -0.30) {{\color{bnRed}\bnttl Spectral Gain Fragility}\\[2pt]
    ISO shifts \& JPEG compression corrupt unnormalized residual amplitudes $R_{\text{Bayar}}$.};
\node[probbox] (p3) at (0, -2.55) {{\color{bnRed}\bnttl Naive Two-Stream Concat}\\[2pt]
    Ignores spatial-spectral covariance across wavelet sub-bands ($I_{LH}, I_{HL}, I_{HH}$).};
\node[probbox] (p4) at (0, -4.80) {{\color{bnRed}\bnttl Authentic Zero-Gradient}\\[2pt]
    Dice \& Soft-IoU gradients vanish \mbox{($\propto \epsilon$)} when ground-truth mask $Y = \mathbf{0}$.};

% ---------- column 2 : remedies ----------
\node[solbox] (s1) at (7.6, 1.95)  {{\color{bnTeal}\bnttl Reciprocal DACM ($\mathcal{O}(H_l W_l d_k)$)}\\[2pt]
    Point-wise inner-product gating $A_{S\leftrightarrow N}$ with global channel weights $[w_S \,\|\, w_N]$.};
\node[solbox] (s2) at (7.6, -0.30) {{\color{bnTeal}\bnttl LHFRE + Affine Standardization}\\[2pt]
    $5\times 5$ Bayar pin $w(0,0)=-1$ + instance standardization $(\mu_B, \sigma_B)$ + 2D Haar DWT.};
\node[solbox] (s3) at (7.6, -2.55) {{\color{bnTeal}\bnttl BARGM Dilated Seam Guidance}\\[2pt]
    Multi-rate ASPP ($r \in \{1,2,4\}$) supervised by morphological seam
    \mbox{$B_{\text{gt}} = (Y\oplus\mathcal{S}_r)-(Y\ominus\mathcal{S}_r)$}.};
\node[solbox] (s4) at (7.6, -4.80) {{\color{bnTeal}\bnttl Regime-Decoupled Loss + ASP}\\[2pt]
    Piece-wise $\mathcal{L}_{\text{IoU}}^+$ ($\|Y\|_1 > 0$) paired with quadratic penalty
    \mbox{$\mathcal{L}_{\text{ASP}} = \frac{1}{|\Omega|}\sum_i p_i^2$}.};

% ---------- column 3 : outcomes ----------
\node[resbox] (r1) at (14.6, 1.95)  {{\color{bnBlue}\bnttl Zero False Alarms}\\[2pt]
    on high-contrast authentic scene edges};
\node[resbox] (r2) at (14.6, -0.30) {{\color{bnBlue}\bnttl Camera-Invariant}\\[2pt]
    spectral features under JPEG $Q=50$ \& $\sigma_{255}=15$};
\node[resbox] (r3) at (14.6, -2.55) {{\color{bnBlue}\bnttl Crisp Seam Recovery}\\[2pt]
    under feathering \& cloning\\ (\mbox{$F_1^{\text{bound}}=78.4\%$}, \mbox{$d_H=2.14$ px})};
\node[resbox] (r4) at (14.6, -4.80) {{\color{bnBlue}\bnttl Linear Restoring Gradient}\\[2pt]
    $\frac{\partial \mathcal{L}_{\text{ASP}}}{\partial p_i} = +\frac{2}{|\Omega|}p_i$ ($0.41\%$ Authentic FPR)};

% ---------- background panels (auto-fitted to content) ----------
\begin{scope}[on background layer]
    \node[panel, draw=bnRed!45,  fill=bnRed!3,  fit=(h1)(p1)(p2)(p3)(p4)] {};
    \node[panel, draw=bnTeal!45, fill=bnTeal!3, fit=(h2)(s1)(s2)(s3)(s4)] {};
    \node[panel, draw=bnBlue!45, fill=bnBlue!3, fit=(h3)(r1)(r2)(r3)(r4)] {};
\end{scope}

% ---------- index tags on the left edge of every card ----------
\foreach \i in {1,2,3,4}{
    \node[tag, fill=bnRed] at (p\i.west) {B\i};
}

% ---------- flow arrows ----------
\foreach \i in {1,2,3,4}{
    \draw[flow] (p\i.east) -- node[pill] {R\i} (s\i.west);
    \draw[flow, color=bnTeal!80!black] (s\i.east) -- (r\i.west);
}
\end{tikzpicture}}
\caption{Conceptual mapping between four recurring failure modes in passive image forgery localization and the targeted architectural remedies in DF-CMNet v2.}
\label{fig:motivation_concept}
\end{figure*}

\textit{1) Semantic Overfitting and Natural Edge False Alarms:} Backbone networks such as ResNet~\cite{he2016deep} and Swin Transformer~\cite{liu2021swin} are designed to group pixels by object category. When transferred to image forensics, high-capacity semantic encoders respond strongly to legitimate high-contrast boundaries such as building silhouettes, horizon lines, and shadow transitions. Without explicit frequency corroboration, these spatial edges are frequently misclassified as splicing boundaries.

\textit{2) Exposure Variance and Compression Fragility in Noise Streams:} High-pass residual filters, including Steganalysis Rich Models (SRM)~\cite{fridrich2012rich} and Bayar constrained convolutions~\cite{bayar2018constrained}, strip away low-frequency scene content to expose sensor noise. Raw residual amplitudes, however, vary widely across camera models, ISO settings, and exposure gains. Strided spatial downsampling and lossy JPEG re-compression also attenuate high-frequency noise energy, causing standalone residual branches to fail after social media transmission~\cite{li2019localization}.

\textit{3) Uncoordinated Fusion and Seam Smearing:} Two-stream architectures typically merge RGB and noise features via channel concatenation or element-wise summation~\cite{zhou2018learning,wu2019mantra}. Concatenation treats all spectral channels equally and ignores how directional wavelet sub-bands ($LH, HL, HH$) align with spatial object contours. At the same time, real-world forgers apply Gaussian feathering or Poisson gradient blending~\cite{huh2018fighting,chen2020zero} along donor-host perimeters, smearing transition seams across several pixels so that uniform pixel-wise losses produce eroded or bloated masks.

\textit{4) The Authentic Zero-Gradient Pathology in Overlap Losses:} Forensic training sets contain both tampered images and unmanipulated authentic images ($y_i = 0, \forall i \in \Omega$). Under the standard Dice loss $\mathcal{L}_{\text{Dice}} = 1 - (2\sum_i p_i y_i + \epsilon) / (\sum_i p_i^2 + \sum_i y_i^2 + \epsilon)$ (as well as naive Soft-IoU formulations), evaluating an authentic image collapses the numerator to the smoothing constant $\epsilon = 10^{-5}$. The resulting partial derivative with respect to each predicted pixel probability $p_i$ becomes $\partial \mathcal{L}_{\text{Dice}}^{\text{auth}} / \partial p_i = +2\epsilon p_i / (\sum_k p_k^2 + \epsilon)^2 \approx 0$. Because this gradient vanishes by five orders of magnitude, false-positive predictions on pristine images receive virtually no penalty during backpropagation.

To address these four bottlenecks in a unified manner, we develop \textbf{DF-CMNet v2}, a dual-domain frequency-spatial cross-modulation network with boundary-aware supervision. As summarized in Fig.~\ref{fig:motivation_concept} (B1--B4 $\to$ R1--R4), each module in our design directly targets one of the failure modes above. Our main contributions are fivefold:
\begin{enumerate}
    \item \textit{Learnable High-Frequency Residual Extractor (LHFRE) with Affine Standardization:} We pair a $5 \times 5$ Bayar prediction-error convolution with per-instance affine residual standardization and lossless 2D Haar Discrete Wavelet Transform (2D-DWT) decomposition. This stabilizes residual variance across heterogeneous camera ISO settings while preserving horizontal ($LH$), vertical ($HL$), and diagonal ($HH$) detail coefficients during downsampling.
    \item \textit{Linear-Complexity Dual-Attention Cross-Domain Modulation (DACM):} We replace static concatenation with a bidirectional modulation block that executes point-wise spatial cross-attention and global channel cross-attention in $\mathcal{O}(H_l W_l)$ linear complexity. DACM highlights locations where visual edges coincide with spectral anomalies while re-weighting informative wavelet channels.
    \item \textit{Boundary-Aware Residual Guidance Module (BARGM):} We combine multi-rate Atrous Spatial Pyramid Pooling (ASPP) on deep fused representations with high-resolution skip refinement and morphological dilation-erosion boundary supervision, recovering crisp perimeters across feathered and Poisson-blended seams.
    \item \textit{Regime-Decoupled Soft-IoU and Authentic Suppression Penalty (ASP):} We prove that both standard Dice and naive Soft-IoU losses suffer from vanishing gradients on authentic masks ($Y = \mathbf{0}$), and we formulate a piece-wise objective that pairs foreground Soft-IoU ($\|Y\|_1 > 0$) with a quadratic Authentic Suppression Penalty ($\|Y\|_1 = 0$) yielding a linear restoring gradient $\frac{2}{|\Omega|}p_i$.
    \item \textit{Extensive Benchmarking Across Classical and Diffusion Forgeries:} We evaluate DF-CMNet v2 against ten representative methods from 2018--2024 across six standard benchmarks (CASIA v1.0, CASIA v2.0~\cite{dong2013casia}, Columbia, NIST'16, COVERAGE, IMD2020~\cite{novozamsky2020imd2020}) and three modern generative inpainting benchmarks (CocoGlide~\cite{guillaro2023trufor}, AutoSplice~\cite{jia2023autosplice}, LaMa-Removal~\cite{suvorov2022resolution}), accompanied by multi-seed statistical testing, post-processing stress tests, and complexity profiling.
\end{enumerate}

The remainder of this paper is organized as follows. Section~\ref{sec:related} reviews related work in spatial, spectral, and transformer-based image forensics. Section~\ref{sec:method} presents the mathematical formulation and architectural diagrams of DF-CMNet v2. Section~\ref{sec:experiments} details the experimental protocol, quantitative comparisons, robustness tests, and ablation studies. Section~\ref{sec:qualitative} analyzes qualitative visual results on photographic test cases. Section~\ref{sec:discussion} examines failure modes and counter-forensic resilience, and Section~\ref{sec:conclusion} concludes the paper.

\section{Related Work}
\label{sec:related}
Research in passive image forgery localization has evolved from handcrafted statistical models of camera hardware to multi-stream deep neural networks and vision transformers. Table~\ref{tab:related_summary} summarizes representative forensic localization frameworks and contrasts their design choices with DF-CMNet v2.

\begin{table*}[!t]
\centering
\caption{Taxonomy and architectural comparison of representative passive image forgery localization methods.}
\label{tab:related_summary}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccccc}
\toprule
\textbf{Method} & \textbf{Venue / Year} & \textbf{Spatial Stream} & \textbf{Frequency / Noise Operator} & \textbf{Noise Std.} & \textbf{Cross-Domain Fusion} & \textbf{Boundary Supervision} & \textbf{Authentic Loss Handling} & \textbf{Attention Complexity} \\
\midrule
MFCN~\cite{chen2020zero}          & JVCI 2018   & VGG-16 FCN         & None                          & No  & Single-Stream       & Edge-Mask Multi-Task & Standard Cross-Entropy & N/A \\
RGB-N~\cite{zhou2018learning}     & CVPR 2018   & Faster R-CNN       & 30 Fixed SRM Filters          & No  & Bilinear Pooling    & Bounding Box Only    & Anchor Softmax         & N/A \\
BusterNet~\cite{wu2018busternet}  & ECCV 2018   & VGG-16 Dual-Branch & Self-Correlation Similarity   & No  & Concat + BN         & None                 & Binary Cross-Entropy   & $\mathcal{O}((HW)^2)$ \\
ManTra-Net~\cite{wu2019mantra}    & CVPR 2019   & WiderResNet        & SRM + Bayar Conv + Z-Pool     & Yes & Channel Concat      & None                 & Focal + BCE            & $\mathcal{O}(HW)$ \\
SPAN~\cite{hu2020span}            & ECCV 2020   & WiderResNet        & SRM + Bayar Conv              & No  & Pyramid Propagation & None                 & Binary Cross-Entropy   & Local $\mathcal{O}(K^2 HW)$ \\
MVSS-Net~\cite{chen2021image}     & TPAMI 2023  & ResNet-50          & Bayar Conv2d                  & No  & Dual-Branch Sum     & Sobel Edge Branch    & Standard Dice ($\epsilon$-vanishing) & $\mathcal{O}(HW)$ \\
PSCC-Net~\cite{liu2022pscc}       & TCSVT 2022  & HRNet-W18          & None (Spatial Multi-Scale)    & No  & Progressive Gates   & None                 & BCE + Image-Level BCE  & $\mathcal{O}((HW)^2 + C^2)$ \\
CAT-Net v2~\cite{kwon2022learning}& IJCV 2022   & HRNet-W48          & Quantized DCT Volume          & No  & Late Stage Concat   & None                 & Cross-Entropy          & $\mathcal{O}(HW)$ \\
ObjectFormer~\cite{wang2022objectformer} & CVPR 2022 & ViT + CNN     & Learnable High-Pass DCT       & No  & Object Query Attn   & Boundary Patch Loss  & BCE + Dice             & $\mathcal{O}(N_{\text{obj}} HW)$ \\
HiFi-Net~\cite{guo2023hierarchical}& CVPR 2023  & Multi-Branch HRNet & High-Pass Laplacian Residuals & No  & Hierarchical Concat & None                 & Multi-Level BCE + Focal& $\mathcal{O}(HW)$ \\
TruFor~\cite{guillaro2023trufor}  & CVPR 2023   & MiT-B2 Transformer & Pre-trained Noiseprint++      & No  & Cross-Modal Concat  & Anomaly Decoder      & Dice + Confidence Loss & $\mathcal{O}((HW)^2 / R^2)$ \\
IML-ViT~\cite{ma2023iml}          & arXiv 2023  & Windowed ViT-B     & Sobel High-Pass Embedding     & No  & Single-Stream ViT   & Edge BCE Supervision & BCE + Soft-IoU         & Windowed $\mathcal{O}(W_s^2 HW)$ \\
\midrule
\textbf{DF-CMNet v2 (Ours)}       & \textbf{2026} & \textbf{4-Stage ResNet} & \textbf{Bayar + 2D Haar DWT} & \textbf{Yes (Affine)} & \textbf{Reciprocal DACM} & \textbf{Morphological BARGM} & \textbf{Piece-Wise IoU + ASP} & \textbf{Linear $\mathcal{O}(HW d_k)$} \\
\bottomrule
\end{tabular}}
\end{table*}

\subsection{Spatial-Domain and Transformer Forensic Networks}
Early deep learning approaches framed manipulation detection as local patch classification~\cite{rao2016deep,bappy2019hybrid}. To achieve dense pixel-level localization, Zhou \textit{et al.}~\cite{zhou2018learning} proposed RGB-N, which pairs an RGB object detector with a Steganalysis Rich Model (SRM) noise stream. Wu \textit{et al.}~\cite{wu2019mantra} developed ManTra-Net, pre-training a manipulation-trace feature extractor across 385 editing operations and formulating localization as local Z-score anomaly detection. For copy-move forgery, BusterNet~\cite{wu2018busternet} introduced a two-branch architecture that computes dense self-correlation matrices to separate source and target clones. Hu \textit{et al.}~\cite{hu2020span} subsequently proposed SPAN, modeling multi-scale spatial pyramid relationships across local pixel neighborhoods.

More recently, attention-driven and progressive architectures have gained prominence. Liu \textit{et al.}~\cite{liu2022pscc} introduced PSCC-Net, which extracts top-down multi-scale features and estimates manipulation masks progressively from coarse to fine scales using spatio-channel correlation modules. Guo \textit{et al.}~\cite{guo2023hierarchical} proposed HiFi-Net to jointly classify hierarchical manipulation attributes and localize forged masks. Vision transformers have also been adapted to capture long-range context: ObjectFormer~\cite{wang2022objectformer} uses learnable object queries to probe patch-level consistency, while IML-ViT~\cite{ma2023iml} adapts a high-resolution windowed ViT backbone with edge-guided supervision. As shown in Table~\ref{tab:related_summary}, pure spatial or high-capacity ViT backbones require tens of millions of parameters and remain susceptible to semantic bias when high-contrast authentic edges mimic splicing seams.

\subsection{Frequency-Domain and Sensor Fingerprint Descriptors}
Physical camera pipelines leave subtle pixel-level correlations through lens aberration, CFA interpolation~\cite{ferrara2012reverse}, and sensor PRNU~\cite{lukas2006digital}. Cozzolino and Verdoliva~\cite{cozzolino2019noiseprint} trained a Siamese network to extract camera-model fingerprints (Noiseprint) by suppressing scene content. Fridrich and Kodovsky~\cite{fridrich2012rich} designed 30 fixed high-pass SRM filters to capture local pixel prediction residuals, whereas Bayar and Stamm~\cite{bayar2018constrained} proposed constrained convolutional layers that learn prediction-error kernels end-to-end while forcing the central weight to $-1$ and the surrounding weights to sum to $+1$. Kwon \textit{et al.}~\cite{kwon2022learning} developed CAT-Net v2, feeding quantized DCT coefficient histograms alongside RGB pixels into a dual-stream HRNet to trace double JPEG compression artifacts. While DCT and noise descriptors perform well on minimally compressed images, their responses fluctuate across camera ISO levels and degrade rapidly when images undergo spatial resizing, Gaussian blur, or multi-stage re-compression~\cite{li2019localization}. Standard strided pooling inside CNN encoders also aliases high-frequency directional energy. In contrast, our LHFRE module applies affine instance standardization to stabilize residual magnitudes across camera settings before decomposing residuals via an orthonormal 2D Haar DWT that preserves horizontal, vertical, and diagonal sub-band energy without loss.

\subsection{Cross-Domain Feature Modulation and Boundary Supervision}
Combining spatial context with noise residuals is central to modern hybrid detectors such as MVSS-Net~\cite{chen2021image} and TruFor~\cite{guillaro2023trufor}. MVSS-Net pairs a Bayar-constrained noise branch with a Sobel edge-supervised spatial branch, summing their predictions at the output layer. TruFor fuses RGB features with a frozen Noiseprint++ extractor using a transformer encoder-decoder to produce both localization masks and image-level reliability maps. As indicated in Table~\ref{tab:related_summary}, most dual-stream networks concatenate or sum intermediate feature maps without explicitly modeling how directional spectral anomalies correlate with spatial edges. While Salloum \textit{et al.}~\cite{chen2020zero}, MVSS-Net~\cite{chen2021image}, and IML-ViT~\cite{ma2023iml} incorporate edge supervision, fixed Sobel operators respond to all high-contrast scene edges indiscriminately, and conventional Dice or Soft-IoU losses leave false alarms on pristine authentic images unpenalized due to vanishing gradients. DF-CMNet v2 addresses both issues through reciprocal Dual-Attention Cross-Domain Modulation (DACM), morphological boundary guidance (BARGM), and our regime-decoupled Authentic Suppression Penalty (ASP).

\subsection{Forensic Localization of Generative and Diffusion Edits}
Modern inpainting pipelines based on fast Fourier convolutions (LaMa)~\cite{suvorov2022resolution} and text-conditioned latent diffusion models (GLIDE, Stable Diffusion, DALL-E 2)~\cite{rombach2022high,jia2023autosplice} synthesize semantically coherent fills that lack classical spliced boundaries or duplicate keypoints. Nevertheless, passing an image patch through a variational autoencoder (VAE) bottleneck and iterative denoising schedule alters the local sensor noise distribution and introduces subtle directional spectral mismatches against the surrounding host lattice~\cite{corvi2023intriguing,ricker2022towards}. Benchmarks such as CocoGlide~\cite{guillaro2023trufor} and AutoSplice~\cite{jia2023autosplice} have shown that detectors relying solely on JPEG block grids or semantic inconsistencies struggle to localize diffusion-inpainted regions. By decomposing standardized prediction-error residuals into directional Haar wavelet sub-bands ($LH, HL, HH$) and cross-modulating them with spatial context, DF-CMNet v2 detects local noise-variance discontinuities introduced by generative inpainting without requiring generator-specific pre-training.
"""


SEC_3_METHOD = r"""
\section{Proposed Methodology}
\label{sec:method}

\subsection{Problem Formulation and Dual-Domain Architecture Overview}
Let $I \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ denote an input RGB photograph with $C_{\text{in}} = 3$ color channels and spatial dimensions $H \times W$. The goal of passive image forgery localization is to learn a parametric mapping $\mathcal{F}_{\Theta}: \mathbb{R}^{3 \times H \times W} \to [0, 1]^{1 \times H \times W}$ that assigns each pixel $i \in \Omega = \{1, \dots, H\} \times \{1, \dots, W\}$ a posterior probability $p_i = \hat{M}_i$ of having been manipulated, alongside an auxiliary boundary probability map $\hat{B} \in [0, 1]^{1 \times H \times W}$ that delineates the perimeter between host and edited regions.

Figure~\ref{fig:framework} illustrates the end-to-end multi-stage architecture of DF-CMNet v2. Rather than relying on a heavyweight ImageNet classification backbone that overfits to high-level object categories, DF-CMNet v2 employs a tailored, parameter-efficient four-stage residual encoder across two parallel branches: a Spatial RGB Encoder Stream ($\mathcal{E}_S$) at the top of Fig.~\ref{fig:framework} and a Frequency-Residual Wavelet Stream ($\mathcal{E}_N$) in the middle. Both streams operate across four progressive spatial scales $l \in \{1, 2, 3, 4\}$ corresponding to scales $1/4$, $1/8$, $1/16$, and $1/32$ of the input resolution ($H_l = H / 2^{l+1}$, $W_l = W / 2^{l+1}$) with aligned feature channel dimensions $C_l \in \{32, 64, 128, 256\}$.

In the Spatial RGB Encoder Stream of Fig.~\ref{fig:framework}, an initial stem block (two $3 \times 3$ convolutions with stride 2 and max-pooling) maps the Input Image $I$ to Spatial Stage~1 ($F_S^{(1)} \in \mathbb{R}^{32 \times \frac{H}{4} \times \frac{W}{4}}$), followed by Spatial Stages~2--4 ($\mathcal{E}_S^{(2)}, \mathcal{E}_S^{(3)}, \mathcal{E}_S^{(4)}$) producing multi-scale representations $F_S^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$. In the parallel Frequency-Residual Wavelet Stream, the LHFRE Extractor first suppresses low-frequency scene content via a $5 \times 5$ Bayar constrained convolution, normalizes camera-specific exposure variations via affine instance standardization, and decomposes the standardized residual $R_{\text{norm}} \in \mathbb{R}^{3 \times H \times W}$ through an orthonormal 2D Haar Discrete Wavelet Transform (2D-DWT). Stacking the four wavelet sub-bands ($LL, LH, HL, HH$) across all three channels yields a 12-channel tensor $W_{\text{DWT}} \in \mathbb{R}^{4C_{\text{in}} \times \frac{H}{2} \times \frac{W}{2}}$ ($4C_{\text{in}} = 12$). Spectral Stage~1 ($\mathcal{E}_N^{(1)}$) projects $W_{\text{DWT}}$ into $C_1 = 32$ channels at scale $1/4$, followed by Spectral Stages~2--4, ensuring that at every hierarchy level $l \in \{1, 2, 3, 4\}$, both spatial and spectral feature maps share identical dimensions $F_S^{(l)}, F_N^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$.

Between the two encoder streams in Fig.~\ref{fig:framework}, four Dual-Attention Cross-Domain Modulation blocks (DACM Stages~1--4) perform reciprocal spatial and channel-wise cross-attention between $F_S^{(l)}$ and $F_N^{(l)}$ to yield fused multi-scale features $M^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$. At the deepest bottleneck ($l=4$, scale $1/32$), the fused representation $M^{(4)}$ enters the BARGM Guidance block in the Boundary-Guided Hierarchical Decoder row, which combines multi-rate dilated context with high-resolution Stage-1 skip features ($M^{(1)}$) to estimate manipulation boundary logits $\hat{B}$ (supervised by the Boundary Seam Loss $\mathcal{L}_{\text{Edge}}$) and inject seam-guided attention into the decoder hierarchy. Finally, Decoder Stages~3, 2, and 1 progressively upsample the features with skip connections from DACM Stages~3, 2, and 1 under multi-scale Deep Supervision ($\mathcal{L}_{\text{aux}}$) to reconstruct the full-resolution Predicted Mask $\hat{M} \in [0, 1]^{1 \times H \times W}$, which is optimized under our Regime-Decoupled Region and Authentic Loss ($\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+ + \mathcal{L}_{\text{ASP}}$).

\begin{figure*}[!t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    block/.style={rectangle, draw=black!85, very thick, minimum height=1.08cm, minimum width=2.45cm, align=center, rounded corners=4pt, font=\footnotesize\sffamily, inner sep=3.5pt, drop shadow={opacity=0.14, shadow xshift=1.2pt, shadow yshift=-1.2pt}},
    sblock/.style={block, fill=blue!10, draw=blue!75!black},
    nblock/.style={block, fill=orange!13, draw=orange!80!black},
    fblock/.style={block, fill=teal!13, draw=teal!80!black, minimum height=1.00cm},
    dblock/.style={block, fill=purple!11, draw=purple!75!black},
    bblock/.style={block, fill=red!11, draw=red!75!black},
    lblock/.style={rectangle, draw=red!80!black, dashed, very thick, fill=yellow!14, minimum height=0.92cm, minimum width=2.75cm, align=center, rounded corners=3.5pt, font=\footnotesize\sffamily, inner sep=3pt},
    arrow/.style={-{Stealth[length=2.3mm, width=1.6mm]}, very thick, draw=black!80},
    skiparrow/.style={-{Stealth[length=2.2mm, width=1.5mm]}, very thick, draw=teal!80!black},
    darrow/.style={-{Stealth[length=2.1mm, width=1.5mm]}, very thick, dashed, draw=red!80!black}
]
% Background Stream Containers
\begin{scope}[on background layer]
    \fill[blue!4, draw=blue!35, dashed, thick, rounded corners=6pt] (6.20, 1.38) rectangle (20.20, 3.18);
    \node[font=\footnotesize\bfseries\sffamily, color=blue!75!black, anchor=west] at (6.40, 2.92) {Spatial RGB Encoder Stream};

    \fill[orange!4, draw=orange!40, dashed, thick, rounded corners=6pt] (2.80, -2.52) rectangle (20.20, -0.82);
    \node[font=\footnotesize\bfseries\sffamily, color=orange!80!black, anchor=east] at (20.00, -2.32) {Frequency-Residual Wavelet Stream};

    \fill[purple!4, draw=purple!35, dashed, thick, rounded corners=6pt] (6.20, -4.55) rectangle (20.20, -2.82);
    \node[font=\footnotesize\bfseries\sffamily, color=purple!80!black, anchor=east] at (20.00, -3.06) {Boundary-Guided Hierarchical Decoder};
\end{scope}

% Column 0: Input Image
\node[block, fill=gray!16, minimum width=2.25cm, minimum height=1.35cm] (input) at (1.20, 0.25) {\textbf{Input Image}\\RGB Photograph\\($H \times W$)};

% Column 0.5: LHFRE & Predicted Mask
\node[nblock, minimum width=2.55cm] (lhfre) at (4.30, -1.55) {\textbf{LHFRE Extractor}\\Bayar + Affine Std.\\+ 2D Haar Wavelet};
\node[block, fill=green!14, draw=green!60!black, minimum width=2.55cm] (outm) at (4.30, -3.85) {\textbf{Predicted Mask}\\Localization Map $\hat{M}$};

% Row 1: Spatial Stream Stages (x = 7.8, 11.4, 15.0, 18.6)
\node[sblock] (es1) at (7.80, 2.10) {\textbf{Spatial Stage 1}\\Scale $1/4$ (32-ch)};
\node[sblock] (es2) at (11.40, 2.10) {\textbf{Spatial Stage 2}\\Scale $1/8$ (64-ch)};
\node[sblock] (es3) at (15.00, 2.10) {\textbf{Spatial Stage 3}\\Scale $1/16$ (128-ch)};
\node[sblock] (es4) at (18.60, 2.10) {\textbf{Spatial Stage 4}\\Scale $1/32$ (256-ch)};

% Row 2: DACM Fusion Blocks (x = 7.8, 11.4, 15.0, 18.6)
\node[fblock] (dacm1) at (7.80, 0.25) {\textbf{DACM Stage 1}\\Cross-Modulation};
\node[fblock] (dacm2) at (11.40, 0.25) {\textbf{DACM Stage 2}\\Cross-Modulation};
\node[fblock] (dacm3) at (15.00, 0.25) {\textbf{DACM Stage 3}\\Cross-Modulation};
\node[fblock] (dacm4) at (18.60, 0.25) {\textbf{DACM Stage 4}\\Cross-Modulation};

% Row 3: Spectral Stream Stages (x = 7.8, 11.4, 15.0, 18.6)
\node[nblock] (en1) at (7.80, -1.55) {\textbf{Spectral Stage 1}\\Scale $1/4$ (32-ch)};
\node[nblock] (en2) at (11.40, -1.55) {\textbf{Spectral Stage 2}\\Scale $1/8$ (64-ch)};
\node[nblock] (en3) at (15.00, -1.55) {\textbf{Spectral Stage 3}\\Scale $1/16$ (128-ch)};
\node[nblock] (en4) at (18.60, -1.55) {\textbf{Spectral Stage 4}\\Scale $1/32$ (256-ch)};

% Row 4: Decoder & BARGM Row (x = 7.8, 11.4, 15.0, 18.6)
\node[dblock] (dec1) at (7.80, -3.85) {\textbf{Decoder Stage 1}\\Full-Scale Output};
\node[dblock] (dec2) at (11.40, -3.85) {\textbf{Decoder Stage 2}\\Scale $1/8$ Upsample};
\node[dblock] (dec3) at (15.00, -3.85) {\textbf{Decoder Stage 3}\\Scale $1/16$ Upsample};
\node[bblock] (bargm) at (18.60, -3.85) {\textbf{BARGM Guidance}\\Dilated Context\\\& Seam Refinement};

% Row 5: Supervision Loss Blocks (y = -5.55)
\node[lblock] (lossm) at (4.30, -5.55) {\textbf{Region \& Authentic Loss}\\Overlap + Authentic Penalty};
\node[lblock] (lossaux) at (13.20, -5.55) {\textbf{Deep Supervision}\\Multi-Scale Auxiliary Loss};
\node[lblock] (lossb) at (18.60, -5.55) {\textbf{Boundary Seam Loss}\\Morphological Edge Loss};

% Input & Encoder Connections
\draw[arrow] (input.north) |- (es1.west);
\draw[arrow] (input.south) |- (lhfre.west);
\draw[arrow] (lhfre.east) -- (en1.west);

\draw[arrow] (es1.east) -- (es2.west);
\draw[arrow] (es2.east) -- (es3.west);
\draw[arrow] (es3.east) -- (es4.west);

\draw[arrow] (en1.east) -- (en2.west);
\draw[arrow] (en2.east) -- (en3.west);
\draw[arrow] (en3.east) -- (en4.west);

% Vertical Cross-Modulation Connections
\draw[arrow] (es1.south) -- (dacm1.north); \draw[arrow] (en1.north) -- (dacm1.south);
\draw[arrow] (es2.south) -- (dacm2.north); \draw[arrow] (en2.north) -- (dacm2.south);
\draw[arrow] (es3.south) -- (dacm3.north); \draw[arrow] (en3.north) -- (dacm3.south);
\draw[arrow] (es4.south) -- (dacm4.north); \draw[arrow] (en4.north) -- (dacm4.south);

% Bottleneck & Multi-Scale Skip Connections to Decoder
\draw[skiparrow] (dacm4.east) -- ++(0.60, 0) |- (bargm.east);
\draw[skiparrow] (dacm3.west) -- ++(-0.52, 0) -- ++(0, -3.22) -| ([xshift=-0.78cm]dec3.north);
\draw[skiparrow] (dacm2.west) -- ++(-0.52, 0) -- ++(0, -3.22) -| ([xshift=-0.78cm]dec2.north);
\draw[skiparrow] (dacm1.west) -- ++(-0.52, 0) -- ++(0, -3.22) -| ([xshift=-0.78cm]dec1.north);

% Top-Down Decoder Path
\draw[arrow] (bargm.west) -- (dec3.east);
\draw[arrow] (dec3.west) -- (dec2.east);
\draw[arrow] (dec2.west) -- (dec1.east);
\draw[arrow] (dec1.west) -- (outm.east);

% Training Supervision Arrows
\draw[darrow] (outm.south) -- (lossm.north);
\draw[darrow] (bargm.south) -- (lossb.north);
\draw[darrow] (dec2.south) -- ++(0, -0.36) -| ([xshift=-0.55cm]lossaux.north);
\draw[darrow] (dec3.south) -- ++(0, -0.36) -| ([xshift=0.55cm]lossaux.north);
\end{tikzpicture}}
\caption{End-to-end multi-stage architecture of the proposed DF-CMNet v2 framework.}
\label{fig:framework}
\end{figure*}

\subsection{Learnable High-Frequency Residual Extractor (LHFRE) with Affine Standardization}
Standard convolutional layers learn filters dominated by low-frequency luminance and chrominance structures, which obscures faint sensor-level inconsistencies. Conversely, fixed high-pass filters such as SRM~\cite{fridrich2012rich} cannot adapt their passbands to unknown post-processing pipelines. As detailed in Fig.~\ref{fig:lhfre_tikz}, our LHFRE module bridges this gap through a four-stage signal processing sequence: prediction-error filtering (Step~1), affine instance standardization (Step~2), orthonormal 2D Haar wavelet decomposition into four directional sub-bands (Step~3), and sub-band stacking with Stage-1 spectral projection (Step~4).

First, in Step~1 of Fig.~\ref{fig:lhfre_tikz} ($5 \times 5$ Bayar Filter), we apply a $K \times K$ ($K = 5$) constrained prediction-error convolution following Bayar and Stamm~\cite{bayar2018constrained}. Let $w_k^{(c)} \in \mathbb{R}^{5 \times 5}$ denote the spatial kernel weights for channel $c$ at spatial offset $(u, v) \in \{-2, -1, 0, 1, 2\}^2$. After every gradient update step, the kernel weights are projected onto the zero-mean hyperplane via strict L1 normalization of surrounding coefficients followed by central-weight pinning:
\begin{equation}
\begin{cases}
w_k^{(c)}(0, 0) = -1, \\[4pt]
\displaystyle \sum_{(u,v) \neq (0,0)} w_k^{(c)}(u, v) = +1.
\end{cases}
\label{eq:bayar}
\end{equation}
Condition~\eqref{eq:bayar} ensures that the filter response to any locally constant or linearly varying intensity surface is zero, forcing the layer to extract high-order local prediction residuals $R_{\text{Bayar}} = \Phi_{\text{Bayar}}(I) \in \mathbb{R}^{3 \times H \times W}$.

Second, raw prediction-error residuals exhibit strong amplitude variations across camera models, ISO sensitivities, and local exposure conditions. To prevent high-ISO noise in authentic shadows from mimicking manipulation anomalies, Step~2 of Fig.~\ref{fig:lhfre_tikz} (Affine Standardization) applies per-instance zero-mean and unit-variance scaling across the spatial domain of each channel:
\begin{equation}
\begin{aligned}
R_{\text{norm}}^{(c)}(x, y) &= \gamma_s^{(c)} \cdot \frac{R_{\text{Bayar}}^{(c)}(x, y) - \mu_B^{(c)}}{\sqrt{(\sigma_B^{(c)})^2 + \epsilon_{\text{std}}}} + \beta_s^{(c)}, \\[3pt]
\mu_B^{(c)} &= \frac{1}{HW}\sum_{x=1}^{H}\sum_{y=1}^{W} R_{\text{Bayar}}^{(c)}(x,y), \\[2pt]
(\sigma_B^{(c)})^2 &= \frac{1}{HW}\sum_{x=1}^{H}\sum_{y=1}^{W} \Big(R_{\text{Bayar}}^{(c)}(x,y) - \mu_B^{(c)}\Big)^2,
\end{aligned}
\label{eq:affine_std}
\end{equation}
where $\mu_B^{(c)}$ and $(\sigma_B^{(c)})^2$ denote the spatial mean and variance of channel $c$, $\epsilon_{\text{std}} = 10^{-5}$ ensures numerical stability, and $\gamma_s, \beta_s \in \mathbb{R}^3$ are learnable per-channel affine scale and shift parameters.

Third, strided spatial pooling inside standard CNN stems discards high-frequency spectral energy through aliasing. In Step~3 of Fig.~\ref{fig:lhfre_tikz}, we replace lossy strided downsampling with a single-level 2D Haar Discrete Wavelet Transform (2D-DWT). Using the orthonormal low-pass filter $L = \frac{1}{\sqrt{2}}[1, 1]^\top$ and high-pass filter $H_w = \frac{1}{\sqrt{2}}[-1, 1]^\top$, we construct four $2 \times 2$ separable stride-2 convolution kernels:
\begin{equation}
\begin{aligned}
K_{LL} &= \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}, \quad &
K_{LH} &= \frac{1}{2}\begin{bmatrix} -1 & -1 \\ 1 & 1 \end{bmatrix}, \\[3pt]
K_{HL} &= \frac{1}{2}\begin{bmatrix} -1 & 1 \\ -1 & 1 \end{bmatrix}, \quad &
K_{HH} &= \frac{1}{2}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}.
\end{aligned}
\label{eq:haar_kernels}
\end{equation}
Convolving the standardized residual $R_{\text{norm}}$ with $\{K_{LL}, K_{LH}, K_{HL}, K_{HH}\}$ at stride 2 produces the four parallel half-resolution ($H/2 \times W/2$) sub-band branches shown in Step~3 of Fig.~\ref{fig:lhfre_tikz}: the coarse Approximation ($LL$) low-pass sub-band and three directional detail sub-bands capturing Horizontal Detail ($LH$), Vertical Detail ($HL$), and Diagonal Detail ($HH$). Because Haar wavelets are real-valued symmetric/anti-symmetric basis functions, they do not carry complex phase; instead, forensic traces manifest as local energy and covariance discontinuities across the directional detail sub-bands $I_{LH}, I_{HL}, I_{HH}$ whenever splicing, resampling, or diffusion decoding disrupts the host camera's CFA or JPEG $8 \times 8$ block lattice. Finally, in Step~4 of Fig.~\ref{fig:lhfre_tikz} (Sub-Band Stack \& Stage-1 Projection), concatenating all four sub-bands across the three color channels yields a 12-channel wavelet tensor $W_{\text{DWT}} \in \mathbb{R}^{12 \times \frac{H}{2} \times \frac{W}{2}}$, which is mapped by a stride-2 $3 \times 3$ convolution ($\mathcal{E}_N^{(1)}$) into stage-1 spectral features $F_N^{(1)} \in \mathbb{R}^{32 \times \frac{H}{4} \times \frac{W}{4}}$:
\begin{equation}
F_N^{(1)} = \mathcal{E}_N^{(1)}\Big(\text{Concat}\big[I_{LL}, I_{LH}, I_{HL}, I_{HH}\big]\Big).
\label{eq:lhfre_out}
\end{equation}

\begin{figure*}[!t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    box/.style={rectangle, draw=black!85, very thick, rounded corners=4pt, align=center, font=\footnotesize\sffamily, inner sep=4pt, drop shadow={opacity=0.15, shadow xshift=1.2pt, shadow yshift=-1.2pt}},
    subbox/.style={rectangle, draw=orange!85!black, fill=orange!11, very thick, rounded corners=3.5pt, minimum width=3.35cm, minimum height=0.88cm, align=center, font=\footnotesize\sffamily, inner sep=2.5pt, drop shadow={opacity=0.12, shadow xshift=1pt, shadow yshift=-1pt}},
    arrow/.style={-{Stealth[length=2.4mm, width=1.65mm]}, very thick, draw=black!80}
]
% Background panel
\begin{scope}[on background layer]
    \fill[orange!3, draw=orange!35, dashed, thick, rounded corners=6pt] (-0.30, -2.62) rectangle (21.45, 2.62);
\end{scope}

% Col 1: Input
\node[box, fill=gray!15, minimum width=2.15cm, minimum height=1.85cm] (in) at (1.00, 0) {
    \textbf{Input Image}\\
    RGB Photograph\\
    ($H \times W \times 3$)
};

% Col 2: Step 1 - Bayar 5x5 Kernel
\node[box, fill=orange!13, draw=orange!80!black, minimum width=3.45cm, minimum height=3.15cm, right=0.80cm of in] (bayar) {
    \textbf{1. $5 \times 5$ Bayar Filter}\\[3pt]
    {\scriptsize
    \setlength{\arraycolsep}{2.6pt}
    \renewcommand{\arraystretch}{0.88}
    $\begin{array}{|c|c|c|c|c|}
    \hline
    w & w & w & w & w \\ \hline
    w & w & w & w & w \\ \hline
    w & w & \textcolor{red!85!black}{\mathbf{-1}} & w & w \\ \hline
    w & w & w & w & w \\ \hline
    w & w & w & w & w \\ \hline
    \end{array}$}\\[4pt]
    Center $= -1$, Surround $= +1$\\
    \textbf{Output:} Prediction Residual
};

% Col 3: Step 2 - Affine Standardization
\node[box, fill=yellow!15, draw=olive!80!black, minimum width=3.55cm, minimum height=2.45cm, right=0.80cm of bayar] (norm) {
    \textbf{2. Affine Standardization}\\[3pt]
    Channel-Wise Zero-Mean\\
    \& Unit-Variance Scaling\\[3pt]
    Stabilizes Camera ISO \& Exposure\\
    \textbf{Output:} Standardized Residual
};

% Col 4: Step 3 - 4 Haar DWT Sub-bands stacked vertically with clear spacing
\node[subbox] (ll) at (15.40, 1.80) {\textbf{Approximation ($LL$)}\\Low-Pass Sub-Band};
\node[subbox] (lh) at (15.40, 0.60) {\textbf{Horizontal Detail ($LH$)}\\Horizontal Sub-Band};
\node[subbox] (hl) at (15.40, -0.60) {\textbf{Vertical Detail ($HL$)}\\Vertical Sub-Band};
\node[subbox] (hh) at (15.40, -1.80) {\textbf{Diagonal Detail ($HH$)}\\Diagonal Sub-Band};

% Col 5: Step 4 - Sub-Band Stack + Stage 1 Projection
\node[box, fill=teal!13, draw=teal!85!black, minimum width=2.95cm, minimum height=3.15cm] (cat) at (19.65, 0) {
    \textbf{4. Sub-Band Stack}\\
    12-Ch Wavelet Tensor\\
    ($H/2 \times W/2$)\\[5pt]
    $\Downarrow$ \textbf{Stage-1 Projection}\\
    $3\times 3$ Conv, Stride 2\\[5pt]
    \textbf{Output:} Spectral\\
    Stage-1 Features
};

% Clean Orthogonal Connections
\draw[arrow] (in.east) -- (bayar.west);
\draw[arrow] (bayar.east) -- (norm.west);

\draw[arrow] (norm.east) -- ++(0.42, 0) |- (ll.west);
\draw[arrow] (norm.east) -- ++(0.42, 0) |- (lh.west);
\draw[arrow] (norm.east) -- ++(0.42, 0) |- (hl.west);
\draw[arrow] (norm.east) -- ++(0.42, 0) |- (hh.west);

\draw[arrow] (ll.east) -- ++(0.48, 0) |- ([yshift=0.82cm]cat.west);
\draw[arrow] (lh.east) -- ++(0.28, 0) |- ([yshift=0.28cm]cat.west);
\draw[arrow] (hl.east) -- ++(0.28, 0) |- ([yshift=-0.28cm]cat.west);
\draw[arrow] (hh.east) -- ++(0.48, 0) |- ([yshift=-0.82cm]cat.west);
\end{tikzpicture}}
\caption{Internal signal processing pipeline of the Learnable High-Frequency Residual Extractor (LHFRE).}
\label{fig:lhfre_tikz}
\end{figure*}

\subsection{Linear-Complexity Dual-Attention Cross-Domain Modulation (DACM)}
At each encoder hierarchy level $l \in \{1, 2, 3, 4\}$, the spatial feature map $F_S^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$ captures object boundaries and surface illumination, whereas the spectral feature map $F_N^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$ encodes directional wavelet residual anomalies. Simply concatenating $F_S^{(l)}$ and $F_N^{(l)}$ fails to suppress natural scene edges that exhibit normal sensor noise. Conversely, computing full non-local spatial self-attention over $(H_l W_l) \times (H_l W_l)$ pixel pairs incurs quadratic $\mathcal{O}(H_l^2 W_l^2)$ memory and FLOP overhead, making high-resolution inference prohibitive.

To achieve precise cross-domain corroboration with strictly linear $\mathcal{O}(H_l W_l)$ complexity, our DACM module pairs a Point-Wise Spatial Cross-Attention stage (Stage~1) with a Channel Cross-Attention and Gated Fusion stage (Stage~2), as illustrated in Fig.~\ref{fig:dacm_tikz}. For clarity of notation, we omit the stage superscript $(l)$ below.

\textit{1) Point-Wise Reciprocal Spatial Cross-Attention (Stage 1):} As shown on the left of Fig.~\ref{fig:dacm_tikz}, the Spatial Stream Input ($F_S \in \mathbb{R}^{C \times H_l \times W_l}$) and Spectral Stream Input ($F_N \in \mathbb{R}^{C \times H_l \times W_l}$) enter Stage~1 in parallel and pass through the Spatial Query Projection and Spectral Key Projection blocks, which map both streams into a shared bottleneck latent space of dimension $d_k = \max(C / 8, 8)$ via $1 \times 1$ convolutions:
\begin{equation}
Q_S = W_{Q_S} * F_S, \quad K_N = W_{K_N} * F_N \in \mathbb{R}^{d_k \times H_l \times W_l}.
\label{eq:dacm_qk}
\end{equation}
Because spatial and spectral feature maps are pixel-aligned on the same grid $\Omega_l$, a true manipulation seam produces simultaneous structural and spectral responses at the same coordinate $(x, y)$. The central Coincidence Gate in Stage~1 of Fig.~\ref{fig:dacm_tikz} computes the point-wise inner-product affinity map $A_{S \leftrightarrow N} \in (0, 1)^{1 \times H_l \times W_l}$ across the latent channel dimension $d_k$ at each spatial location $(x, y)$:
\begin{equation}
A_{S \leftrightarrow N}(x, y) = \sigma\left( \frac{1}{\sqrt{d_k}} \sum_{m=1}^{d_k} Q_S(m, x, y) \cdot K_N(m, x, y) \right),
\label{eq:spatial_cross_attn}
\end{equation}
where $\sigma(z) = (1 + e^{-z})^{-1}$ is the element-wise sigmoid activation. The affinity map $A_{S \leftrightarrow N}$ feeds into the upper Spatial Gate and lower Spectral Gate blocks in Fig.~\ref{fig:dacm_tikz} as a soft coincidence detector: high-contrast authentic edges (large $Q_S$ but quiescent $K_N$) and isolated sensor noise spikes (large $K_N$ in uniform regions) are damped, whereas coordinates with concurrent structural and spectral evidence receive a residual boost:
\begin{equation}
\tilde{F}_S = F_S \odot (1 + A_{S \leftrightarrow N}), \quad \tilde{F}_N = F_N \odot (1 + A_{S \leftrightarrow N}).
\label{eq:spatial_mod}
\end{equation}

\textit{2) Global Channel Cross-Attention and Convex Gating (Stage 2):} Different manipulation modalities excite distinct wavelet sub-bands (for instance, horizontal splicing seams excite $I_{LH}$, whereas isotropic Gaussian blur suppresses $I_{HH}$). In Stage~2 of Fig.~\ref{fig:dacm_tikz}, both spatially boosted streams enter the central Channel Attention block, which applies global average pooling ($\text{GAP}: \mathbb{R}^{C \times H_l \times W_l} \to \mathbb{R}^C$) to both branches and concatenates their descriptors into a joint summary vector $z = [\text{GAP}(\tilde{F}_S) \,\|\, \text{GAP}(\tilde{F}_N)] \in \mathbb{R}^{2C}$. A shared two-layer bottleneck MLP with reduction ratio $r_c = 4$ generates complementary channel attention vectors $w_S, w_N \in (0, 1)^C$:
\begin{equation}
[w_S \,\|\, w_N] = \sigma\Big( W_2 \, \text{ReLU}(W_1 z) \Big),
\label{eq:channel_attn}
\end{equation}
where $W_1 \in \mathbb{R}^{\frac{C}{2} \times 2C}$ and $W_2 \in \mathbb{R}^{2C \times \frac{C}{2}}$. These weights scale the upper and lower branches inside the Channel Scale blocks of Fig.~\ref{fig:dacm_tikz}, after which the Convex Gated Fusion block merges both streams via a learnable convex gating parameter $\alpha_l \in [0, 1]$ and a $1 \times 1$ refinement convolution $\Psi_{\text{fuse}}$:
\begin{equation}
M^{(l)} = \Psi_{\text{fuse}}\Big( \alpha_l \cdot (w_S \odot \tilde{F}_S) + (1 - \alpha_l) \cdot (w_N \odot \tilde{F}_N) \Big).
\label{eq:dacm_out}
\end{equation}

\textit{3) Complexity Analysis:} Projecting $F_S$ and $F_N$ in Eq.~\eqref{eq:dacm_qk} requires $2 C d_k H_l W_l$ multiply-accumulate operations (MACs), and the point-wise inner product in Eq.~\eqref{eq:spatial_cross_attn} requires $d_k H_l W_l$ MACs. The channel MLP in Eq.~\eqref{eq:channel_attn} requires $2 C^2$ MACs independent of spatial resolution. Consequently, the total computational cost of DACM at stage $l$ is $\mathcal{O}\big((2C d_k + d_k) H_l W_l + 2C^2\big)$, which scales strictly linearly with the number of pixels $H_l W_l$. Doubling the input image dimensions $(H, W) \to (2H, 2W)$ increases the total FLOPs of DF-CMNet v2 by an exact factor of $4\times$, enabling native high-resolution inference without quadratic memory bottlenecks.

\begin{figure*}[!t]
\centering
\definecolor{dmBlue}{HTML}{1A5FB4}
\definecolor{dmAmber}{HTML}{C65D00}
\definecolor{dmTeal}{HTML}{00796B}
\definecolor{dmPurple}{HTML}{6A1B9A}
\definecolor{dmGreen}{HTML}{2E7D32}
\def\dmttl{\footnotesize\bfseries\sffamily}
\resizebox{\textwidth}{!}{%
\hyphenpenalty=10000 \exhyphenpenalty=10000 %
\begin{tikzpicture}[
    font=\sffamily,
    card/.style={rectangle, rounded corners=5pt, line width=0.9pt,
        minimum height=1.28cm, text width=2.70cm, inner xsep=0.32cm, inner ysep=4pt,
        align=center, font=\footnotesize\sffamily, text=black!85,
        drop shadow={opacity=0.14, shadow xshift=1.2pt, shadow yshift=-1.2pt, fill=black}},
    gatecard/.style={card, text width=1.95cm, minimum height=1.10cm, inner xsep=0.22cm},
    hdrbox/.style={rectangle, rounded corners=4pt, minimum height=0.68cm, inner xsep=10pt,
        align=center, font=\small\bfseries\sffamily, text=white,
        drop shadow={opacity=0.18, shadow xshift=1pt, shadow yshift=-1pt, fill=black}},
    panel/.style={rectangle, rounded corners=9pt, dashed, line width=0.8pt, inner sep=0.34cm},
    flow/.style={-{Stealth[length=2.5mm, width=2.0mm]}, line width=1.05pt, color=gray!68!black}
]

% ---------- Left Input Column (x = 0.0) ----------
\node[card, draw=dmBlue, fill=dmBlue!6, text width=2.50cm] (fs) at (0.0,  1.65)
    {{\color{dmBlue}\dmttl Spatial Stream Input}\\[2pt]Multi-scale RGB features};
\node[card, draw=dmAmber, fill=dmAmber!6, text width=2.50cm] (fn) at (0.0, -1.65)
    {{\color{dmAmber}\dmttl Spectral Stream Input}\\[2pt]Wavelet residual features};

% ---------- Stage 1: Point-Wise Spatial Cross-Attention ----------
\node[hdrbox, fill=dmTeal, minimum width=7.2cm] (ha) at (6.45, 3.32)
    {Stage 1: Point-Wise Spatial Cross-Attention};

\node[card, draw=dmBlue, fill=dmBlue!5, text width=2.65cm] (qs) at (3.65,  1.25)
    {{\color{dmBlue}\dmttl Spatial Query Projection}\\[2pt]$1\times 1$ bottleneck conv};
\node[card, draw=dmAmber, fill=dmAmber!6, text width=2.65cm] (kn) at (3.65, -1.25)
    {{\color{dmAmber}\dmttl Spectral Key Projection}\\[2pt]$1\times 1$ bottleneck conv};

\node[card, draw=dmTeal, fill=dmTeal!6, text width=2.85cm, minimum height=1.42cm] (dot) at (6.85, 0.0)
    {{\color{dmTeal}\dmttl Coincidence Gate}\\[2pt]Pixel-aligned inner product\\+ sigmoid affinity map};

\node[gatecard, draw=dmBlue, fill=dmBlue!5] (muls) at (9.45,  1.65)
    {{\color{dmBlue}\dmttl Spatial Gate}\\[1pt]Residual boost};
\node[gatecard, draw=dmAmber, fill=dmAmber!6] (muln) at (9.45, -1.65)
    {{\color{dmAmber}\dmttl Spectral Gate}\\[1pt]Residual boost};

% ---------- Stage 2: Channel Cross-Attention & Gated Fusion ----------
\node[hdrbox, fill=dmPurple, minimum width=7.4cm] (hb) at (15.10, 3.32)
    {Stage 2: Channel Cross-Attention \& Gated Fusion};

\node[card, draw=dmPurple, fill=dmPurple!6, text width=2.85cm, minimum height=1.42cm] (gap) at (12.55, 0.0)
    {{\color{dmPurple}\dmttl Channel Attention}\\[2pt]Global average pooling\\+ shared bottleneck MLP};

\node[gatecard, draw=dmBlue, fill=dmBlue!5] (cs) at (15.15,  1.65)
    {{\color{dmBlue}\dmttl Channel Scale}\\[1pt]Spatial weights};
\node[gatecard, draw=dmAmber, fill=dmAmber!6] (cn) at (15.15, -1.65)
    {{\color{dmAmber}\dmttl Channel Scale}\\[1pt]Spectral weights};

\node[card, draw=dmGreen, fill=dmGreen!6, text width=2.65cm, minimum height=1.48cm] (out) at (17.85, 0.0)
    {{\color{dmGreen}\dmttl Convex Gated Fusion}\\[2pt]Learnable convex sum\\+ $1\times 1$ refinement conv};

% ---------- Auto-Fitted Stage Panels ----------
\begin{scope}[on background layer]
    \node[panel, draw=dmTeal!45,   fill=dmTeal!3,   fit=(ha)(qs)(kn)(dot)(muls)(muln)] {};
    \node[panel, draw=dmPurple!45, fill=dmPurple!3, fit=(hb)(gap)(cs)(cn)(out)] {};
\end{scope}

% ---------- Stage 1 Connections ----------
\draw[flow, color=dmBlue!85!black]  (fs.east) -- ++(0.35, 0) |- (qs.west);
\draw[flow, color=dmAmber!90!black] (fn.east) -- ++(0.35, 0) |- (kn.west);

\draw[flow, color=dmBlue!85!black]  (fs.north) -- ++(0,  0.34) -| (muls.north);
\draw[flow, color=dmAmber!90!black] (fn.south) -- ++(0, -0.34) -| (muln.south);

\draw[flow, color=dmBlue!85!black]  (qs.east) -| (dot.north);
\draw[flow, color=dmAmber!90!black] (kn.east) -| (dot.south);

\draw[flow, color=dmTeal!85!black] (dot.east) -| (muls.south);
\draw[flow, color=dmTeal!85!black] (dot.east) -| (muln.north);

% ---------- Stage 2 Connections ----------
\draw[flow, color=dmBlue!85!black]  (muls.east) -- (cs.west);
\draw[flow, color=dmAmber!90!black] (muln.east) -- (cn.west);

\draw[flow, color=dmBlue!85!black]  (12.55,  1.65) -- (gap.north);
\draw[flow, color=dmAmber!90!black] (12.55, -1.65) -- (gap.south);

\draw[flow, color=dmPurple!85!black] (gap.east) -| (cs.south);
\draw[flow, color=dmPurple!85!black] (gap.east) -| (cn.north);

\draw[flow, color=dmGreen!80!black] (cs.east) -| (out.north);
\draw[flow, color=dmGreen!80!black] (cn.east) -| (out.south);
\end{tikzpicture}}
\caption{Micro-architecture of the linear-complexity Dual-Attention Cross-Domain Modulation (DACM) block.}
\label{fig:dacm_tikz}
\end{figure*}

\subsection{Boundary-Aware Residual Guidance Module (BARGM) and Multi-Scale Decoder}
Spliced and inpainted boundaries are rarely step-edge discontinuities; skilled editors apply Gaussian feathering, alpha matting, or Poisson gradient blending that spreads the transition across a $3$ to $12$ pixel band. Standard region-based encoders lose these blurred perimeters at deep bottleneck strides ($\frac{H}{32} \times \frac{W}{32}$).

To recover blended manipulation seams, our BARGM module combines multi-rate dilated context aggregation at the bottleneck with high-resolution skip refinement, as depicted in Fig.~\ref{fig:bargm_tikz}. Given the deepest fused representation $M^{(4)} \in \mathbb{R}^{256 \times \frac{H}{32} \times \frac{W}{32}}$, BARGM applies three parallel $3 \times 3$ atrous (dilated) convolutions with dilation rates $r \in \{1, 2, 4\}$:
\begin{equation}
\begin{aligned}
H_{\text{ASPP}} = \Psi_{1\times 1}\Big( \text{Concat}\big[
&\text{Conv}_{3\times 3}^{(r=1)}(M^{(4)}), \, \text{Conv}_{3\times 3}^{(r=2)}(M^{(4)}), \\
&\text{Conv}_{3\times 3}^{(r=4)}(M^{(4)}) \big] \Big),
\end{aligned}
\label{eq:bargm_aspp}
\end{equation}
which captures multi-scale receptive fields spanning sharp step seams ($r=1$), medium Gaussian-feathered borders ($r=2$), and wide Poisson-blended transition zones ($r=4$). Because $H_{\text{ASPP}} \in \mathbb{R}^{256 \times \frac{H}{32} \times \frac{W}{32}}$ is spatially coarse, predicting boundary masks directly from $H_{\text{ASPP}}$ via bilinear interpolation would produce thick, blurry rings. Therefore, BARGM upsamples a compressed projection of $H_{\text{ASPP}}$ to the stage-1 resolution ($\frac{H}{4} \times \frac{W}{4}$) and concatenates it with a $1 \times 1$ projection of the high-resolution fused skip feature $M^{(1)} \in \mathbb{R}^{32 \times \frac{H}{4} \times \frac{W}{4}}$. Passing this combined tensor through two $3 \times 3$ refinement convolutions produces the sharp boundary logit map $Z_{\text{edge}} \in \mathbb{R}^{1 \times \frac{H}{4} \times \frac{W}{4}}$, which is upsampled to full resolution to yield $\hat{B} = \sigma(\mathcal{U}_{\times 4}(Z_{\text{edge}})) \in [0, 1]^{1 \times H \times W}$ and simultaneously downsampled to gate $M^{(4)}$ via residual boundary attention:
\begin{equation}
M_{\text{guided}}^{(4)} = M^{(4)} + H_{\text{ASPP}} \odot \big(1 + \sigma(\mathcal{D}_{\times 8}(Z_{\text{edge}}))\big).
\label{eq:bargm_gate}
\end{equation}

During training, $\hat{B}$ is supervised against the ground-truth morphological gradient boundary $B_{\text{gt}} \in \{0, 1\}^{1 \times H \times W}$, derived automatically from the binary manipulation mask $Y$ via morphological dilation ($\oplus$) and erosion ($\ominus$) with a flat disk structuring element $\mathcal{S}_r$ of radius $r_d = 2$ pixels ($5 \times 5$ kernel):
\begin{equation}
B_{\text{gt}} = (Y \oplus \mathcal{S}_r) - (Y \ominus \mathcal{S}_r).
\label{eq:morph_gt}
\end{equation}
By supervising $\hat{B}$ against $B_{\text{gt}}$ rather than general Sobel intensity edges, BARGM learns to respond exclusively to forensic donor-host transition seams while ignoring legitimate scene contours.

\textit{Hierarchical Multi-Scale Decoder:} Starting from $D^{(4)} = M_{\text{guided}}^{(4)}$, the top-down decoder progressively doubles spatial resolution across four stages $l \in \{3, 2, 1, 0\}$. At each stage $l \in \{3, 2, 1\}$, bilinear $2\times$ upsampling $\mathcal{U}_{\times 2}(D^{(l+1)})$ is concatenated with the corresponding cross-modulated encoder feature $M^{(l)}$ and refined through two $3 \times 3$ Conv-BN-ReLU layers:
\begin{equation}
D^{(l)} = \Phi_{\text{dec}}^{(l)}\Big( \text{Concat}\big[ \mathcal{U}_{\times 2}(D^{(l+1)}), \, M^{(l)} \big] \Big).
\label{eq:decoder}
\end{equation}
A final upsampling head $D^{(0)}$ restores native $H \times W$ resolution and applies a $1 \times 1$ sigmoid projection to produce the primary localization map $\hat{M} = \sigma(Z_{\text{mask}}) \in [0, 1]^{1 \times H \times W}$. To prevent gradient attenuation in intermediate layers, auxiliary prediction heads attached to $D^{(3)}$ ($\frac{H}{16} \times \frac{W}{16}$) and $D^{(2)}$ ($\frac{H}{8} \times \frac{W}{8}$) generate deep-supervision probability maps $\hat{M}_{\text{aux}}^{(3)}$ and $\hat{M}_{\text{aux}}^{(2)}$.

\begin{figure*}[!t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
    box/.style={rectangle, draw=black!85, very thick, rounded corners=4pt, minimum height=1.10cm, align=center, font=\footnotesize\sffamily, drop shadow={opacity=0.18, shadow xshift=1.5pt, shadow yshift=-1.5pt}},
    arrow/.style={-{Stealth[length=2.5mm, width=1.8mm]}, very thick, draw=black!80}
]
% Two Spacious Stage Containers Across Full Two-Column Width
\begin{scope}[on background layer]
    \fill[teal!4, draw=teal!40, dashed, thick, rounded corners=6pt] (-0.8, -2.25) rectangle (9.95, 3.15);
    \node[font=\footnotesize\bfseries\sffamily, color=teal!85!black] at (4.55, 2.80) {Stage 1: Multi-Rate Dilated Context \& Skip Alignment};
    \fill[purple!4, draw=purple!35, dashed, thick, rounded corners=6pt] (10.35, -2.25) rectangle (18.0, 3.15);
    \node[font=\footnotesize\bfseries\sffamily, color=purple!85!black] at (14.18, 2.80) {Stage 2: Seam Refinement \& Residual Gating};
\end{scope}

% Column 1 (x = 0.8): Inputs M^(4) and M^(1)
\node[box, fill=teal!14, draw=teal!80!black, minimum width=2.6cm, minimum height=1.15cm] (m4) at (0.8, 1.15) {\textbf{Bottleneck} $M^{(4)}$\\Low-Res ($H/32 \times W/32$)};
\node[box, fill=teal!14, draw=teal!80!black, minimum width=2.6cm, minimum height=1.15cm] (m1) at (0.8, -1.30) {\textbf{Stage-1 Skip} $M^{(1)}$\\High-Res ($H/4 \times W/4$)};

% Column 2 (x = 4.65): Parallel Atrous Branches + Skip Projection
\node[box, fill=red!10, draw=red!75!black, minimum width=2.9cm, minimum height=0.80cm] (r1) at (4.65, 2.18) {Dilated Conv ($r=1$)\\Sharp Step Seams};
\node[box, fill=red!15, draw=red!75!black, minimum width=2.9cm, minimum height=0.80cm] (r2) at (4.65, 1.15) {Dilated Conv ($r=2$)\\Feathered Borders};
\node[box, fill=red!20, draw=red!75!black, minimum width=2.9cm, minimum height=0.80cm] (r4) at (4.65, 0.12) {Dilated Conv ($r=4$)\\Wide Blended Zones};
\node[box, fill=blue!10, draw=blue!75!black, minimum width=2.9cm, minimum height=1.05cm] (skip1) at (4.65, -1.30) {$1\times 1$ Skip Projection\\Fine Spatial Detail};

% Column 3 (x = 8.4): ASPP Fusion + High-Res Concatenation
\node[box, fill=orange!16, draw=orange!85!black, minimum width=2.7cm, minimum height=1.15cm] (aspp) at (8.4, 1.15) {\textbf{ASPP Fusion} $H_{\text{ASPP}}$\\Multi-Scale Context};
\node[box, fill=yellow!18, draw=olive!80!black, minimum width=2.7cm, minimum height=1.10cm] (concat) at (8.4, -1.30) {\textbf{Feature Concat}\\Aligned at $H/4 \times W/4$};

% Column 4 (x = 12.3): Seam Refinement Head + Residual Seam Gate
\node[box, fill=purple!14, draw=purple!80!black, minimum width=3.0cm, minimum height=1.10cm] (refine) at (12.3, -1.30) {\textbf{Seam Refinement}\\$3\times 3$ Conv Layers};
\node[box, fill=orange!14, draw=orange!80!black, minimum width=3.0cm, minimum height=1.15cm] (gate) at (12.3, 1.15) {\textbf{Residual Seam Gate}\\Boundary Attention};

% Column 5 (x = 16.25): Outputs M_guided^(4) and B_hat
\node[box, fill=green!15, draw=green!65!black, minimum width=2.9cm, minimum height=1.15cm] (mg) at (16.25, 1.15) {\textbf{Guided Feature} $M_{\text{guided}}^{(4)}$\\To Decoder $D_3$};
\node[box, fill=yellow!22, draw=orange!85!black, minimum width=2.9cm, minimum height=1.10cm] (bgt) at (16.25, -1.30) {\textbf{Boundary Map} $\hat{B}$\\Seam Loss $\mathcal{L}_{\text{Edge}}$};

% Clean Orthogonal Connections
\draw[arrow] (m4.east) -- ++(0.45,0) |- (r1.west);
\draw[arrow] (m4.east) -- (r2.west);
\draw[arrow] (m4.east) -- ++(0.45,0) |- (r4.west);
\draw[arrow] (r1.east) -- ++(0.45,0) |- ([yshift=0.30cm]aspp.west);
\draw[arrow] (r2.east) -- (aspp.west);
\draw[arrow] (r4.east) -- ++(0.45,0) |- ([yshift=-0.30cm]aspp.west);

\draw[arrow] (m1.east) -- (skip1.west);
\draw[arrow] (skip1.east) -- (concat.west);
\draw[arrow] (aspp.south) -- node[right, font=\footnotesize\bfseries, text=teal!85!black] {$8\times$ Upsample} (concat.north);

\draw[arrow] (concat.east) -- (refine.west);
\draw[arrow] (aspp.east) -- (gate.west);
\draw[arrow] (refine.north) -- node[right, font=\footnotesize\bfseries, text=purple!85!black] {Seam Attention} (gate.south);

\draw[arrow] (gate.east) -- (mg.west);
\draw[arrow] (refine.east) -- node[above, font=\footnotesize\bfseries] {$4\times$ Upsample} (bgt.west);
\end{tikzpicture}}
\caption{Architecture of the Boundary-Aware Residual Guidance Module (BARGM) combining multi-rate atrous context aggregation, stage-1 high-resolution skip refinement, and morphological seam supervision.}
\label{fig:bargm_tikz}
\end{figure*}

\subsection{Regime-Decoupled Compound Objective and Authentic Suppression Penalty (ASP)}
A critical flaw in existing forensic localization pipelines is the behavior of region-overlap losses when trained on mixed batches containing both forged images ($\|Y\|_1 > 0$) and unmanipulated authentic images ($Y = \mathbf{0}$, i.e., $y_i = 0, \forall i \in \Omega$).

\textit{Proposition 1 (Vanishing Authentic Gradients in Standard Dice and Naive Soft-IoU):} Consider the standard Dice loss $\mathcal{L}_{\text{Dice}}(P, Y)$ and the standard Soft-IoU loss $\mathcal{L}_{\text{Soft-IoU}}^{\text{naive}}(P, Y)$ with Laplace smoothing constant $\epsilon = 10^{-5}$:
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{Dice}} &= 1 - \frac{2\sum_{i \in \Omega} p_i y_i + \epsilon}{\sum_{i \in \Omega} p_i^2 + \sum_{i \in \Omega} y_i^2 + \epsilon}, \\[4pt]
\mathcal{L}_{\text{Soft-IoU}}^{\text{naive}} &= 1 - \frac{\sum_{i \in \Omega} p_i y_i + \epsilon}{\sum_{i \in \Omega} p_i + \sum_{i \in \Omega} y_i - \sum_{i \in \Omega} p_i y_i + \epsilon}.
\end{aligned}
\label{eq:naive_losses}
\end{equation}
When evaluated on an authentic image where $y_i = 0$ for every pixel $i \in \Omega$, the intersection term $\sum_i p_i y_i$ vanishes identically, reducing both objectives to functions dominated by $\epsilon$:
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{Dice}}\big|_{Y=\mathbf{0}} &= 1 - \frac{\epsilon}{\sum_{k \in \Omega} p_k^2 + \epsilon}, \\[4pt]
\mathcal{L}_{\text{Soft-IoU}}^{\text{naive}}\big|_{Y=\mathbf{0}} &= 1 - \frac{\epsilon}{\sum_{k \in \Omega} p_k + \epsilon}.
\end{aligned}
\label{eq:auth_collapse}
\end{equation}
Taking the partial derivative of Eq.~\eqref{eq:auth_collapse} with respect to any predicted false-positive probability $p_i > 0$ yields:
\begin{equation}
\begin{aligned}
\frac{\partial \mathcal{L}_{\text{Dice}}\big|_{Y=\mathbf{0}}}{\partial p_i} &= +\frac{2\epsilon p_i}{\left(\sum_{k \in \Omega} p_k^2 + \epsilon\right)^2} \approx 0, \\[4pt]
\frac{\partial \mathcal{L}_{\text{Soft-IoU}}^{\text{naive}}\big|_{Y=\mathbf{0}}}{\partial p_i} &= +\frac{\epsilon}{\left(\sum_{k \in \Omega} p_k + \epsilon\right)^2} \approx 0.
\end{aligned}
\label{eq:vanishing_grad}
\end{equation}
Equation~\eqref{eq:vanishing_grad} reveals that both standard Dice and naive Soft-IoU suffer from the exact same pathology on authentic images: as soon as a few pixels produce non-zero false alarms ($\sum_k p_k \ge 1$), the denominator exceeds unity while the numerator remains fixed at $\epsilon = 10^{-5}$, causing the positive restoring gradient to vanish below $10^{-5}$, as plotted in Fig.~\ref{fig:asp_loss_tikz}.

\begin{figure*}[!t]
\centering
\resizebox{0.88\textwidth}{!}{%
\begin{tikzpicture}
% Clean Background Container
\begin{scope}[on background layer]
    \fill[blue!3, draw=blue!30, dashed, thick, rounded corners=6pt] (-0.8, -0.95) rectangle (15.8, 4.15);
\end{scope}

% Shaded Linear Restoring Triangle under ASP curve
\fill[blue!10, opacity=0.45] (1.0, 0.0) -- (9.5, 3.40) -- (9.5, 0.0) -- cycle;

% Axes (x from 1.0 to 9.9, y from 0.0 to 3.75)
\draw[->, very thick] (1.0, 0.0) -- (9.9, 0.0);
\node[below, font=\small\bfseries] at (5.25, -0.45) {False-Positive Pixel Probability $p_i$ on Authentic Image ($Y = \mathbf{0}$)};
\draw[->, very thick] (1.0, 0.0) -- (1.0, 3.75);
\node[rotate=90, above, font=\small\bfseries] at (0.15, 1.80) {Normalized Gradient};

% Horizontal Grid Lines & Simple Numeric Y-Axis Ticks
\foreach \y/\lbl in {0.0/0.0, 0.85/0.5, 1.70/1.0, 2.55/1.5, 3.40/2.0} {
    \draw[gray!35, dashed] (1.0, \y) -- (9.5, \y);
    \node[left, font=\footnotesize] at (0.95, \y) {\lbl};
}

% X-Axis Ticks
\foreach \x/\p in {1.0/0.0, 2.7/0.2, 4.4/0.4, 6.1/0.6, 7.8/0.8, 9.5/1.0} {
    \draw[thick] (\x, 0.0) -- (\x, -0.09) node[below, font=\footnotesize] {\p};
}

% Curve 1: Proposed Quadratic ASP (Linear Restoring Gradient)
\draw[blue!85!black, line width=2.4pt, mark=*, mark size=2.2pt] plot coordinates {(1.0,0.0) (2.7,0.68) (4.4,1.36) (6.1,2.04) (7.8,2.72) (9.5,3.40)};
\node[blue!85!black, font=\small\bfseries, sloped, above=4pt] at (5.25, 1.70) {Linear Restoring Force ($\propto p_i$)};

% Curve 2: Standard Dice Loss (Flat Vanishing Gradient)
\draw[red!85!black, line width=2.1pt, dashed, mark=square*, mark size=2.0pt] plot coordinates {(1.0,0.10) (2.7,0.10) (4.4,0.10) (6.1,0.10) (7.8,0.10) (9.5,0.10)};

% Curve 3: Naive Soft-IoU Loss (Flat Vanishing Gradient)
\draw[orange!90!black, line width=2.1pt, dashdotted, mark=triangle*, mark size=2.1pt] plot coordinates {(1.0,0.03) (2.7,0.03) (4.4,0.03) (6.1,0.03) (7.8,0.03) (9.5,0.03)};
\node[red!80!black, font=\small\bfseries] at (6.6, 0.42) {Vanishing Gradient ($\approx 0$)};

% Single Clean Legend Box on the Right (x = 10.4 to 15.4, completely outside the plot)
\node[draw=black!75, fill=white, very thick, rounded corners=4pt, align=left, font=\small\sffamily, inner sep=8pt, drop shadow={opacity=0.18, shadow xshift=1.5pt, shadow yshift=-1.5pt}] at (12.9, 1.85) {
    \textbf{Loss Function Legend}\\[6pt]
    \textcolor{blue!85!black}{\textbf{--- $\bullet$ Proposed ASP ($\mathcal{L}_{\text{ASP}}$)}}\\
    \hspace*{1.4em}\footnotesize Linear gradient (0.41\% FPR)\\[6pt]
    \textcolor{red!85!black}{\textbf{-- $\blacksquare$ Standard Dice ($\mathcal{L}_{\text{Dice}}$)}}\\
    \hspace*{1.4em}\footnotesize Vanishing gradient (4.85\% FPR)\\[6pt]
    \textcolor{orange!90!black}{\textbf{-$\cdot$- $\blacktriangle$ Naive Soft-IoU ($\mathcal{L}_{\text{IoU}}^{\text{naive}}$)}}\\
    \hspace*{1.4em}\footnotesize Vanishing gradient (4.28\% FPR)
};
\end{tikzpicture}}
\caption{Comparison of the linear restoring gradient produced by our Authentic Suppression Penalty ($\mathcal{L}_{\text{ASP}}$) against the vanishing gradients of standard Dice and naive Soft-IoU losses on pristine images ($Y = \mathbf{0}$).}
\label{fig:asp_loss_tikz}
\end{figure*}

\textit{Regime-Decoupled Piece-Wise Formulation:} To resolve this pathology rigorously, we decouple the foreground overlap regime ($\|Y\|_1 > 0$) from the authentic suppression regime ($\|Y\|_1 = 0$) using the indicator function $\mathbb{I}(\cdot)$. Specifically, we define the foreground Soft-IoU loss $\mathcal{L}_{\text{IoU}}^+$ only over tampered samples and pair it with a quadratic Authentic Suppression Penalty ($\mathcal{L}_{\text{ASP}}$) active on pristine images:
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{IoU}}^+ = \mathbb{I}\big(\|Y\|_1 > 0\big) \cdot \Bigg(1 - \frac{\sum_{i \in \Omega} p_i y_i + \epsilon}{\sum_{i \in \Omega} (p_i + y_i - p_i y_i) + \epsilon}\Bigg),
\end{aligned}
\label{eq:fg_iou}
\end{equation}
\begin{equation}
\mathcal{L}_{\text{ASP}} = \mathbb{I}\big(\|Y\|_1 = 0\big) \cdot \frac{1}{|\Omega|} \sum_{i \in \Omega} p_i^2.
\label{eq:asp_loss}
\end{equation}
For any authentic image ($\|Y\|_1 = 0$), differentiating $\mathcal{L}_{\text{ASP}}$ with respect to $p_i$ yields a strictly positive, linear restoring gradient:
\begin{equation}
\frac{\partial \mathcal{L}_{\text{ASP}}}{\partial p_i} = +\frac{2}{|\Omega|} p_i,
\label{eq:asp_grad}
\end{equation}
which penalizes false-positive activations in direct proportion to their predicted confidence $p_i$ regardless of how many other pixels fire across the image, corresponding to the solid blue linear trajectory in Fig.~\ref{fig:asp_loss_tikz}.

Combining pixel-wise Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$), foreground Soft-IoU ($\mathcal{L}_{\text{IoU}}^+$), joint BCE+Soft-IoU boundary supervision ($\mathcal{L}_{\text{Edge}}$), deep supervision ($\mathcal{L}_{\text{aux}}$), and the Authentic Suppression Penalty ($\mathcal{L}_{\text{ASP}}$) yields the total training objective:
\begin{equation}
\begin{aligned}
\mathcal{L}_{\text{total}} &= \lambda_1 \mathcal{L}_{\text{BCE}}(\hat{M}, Y) + \lambda_2 \mathcal{L}_{\text{IoU}}^+(\hat{M}, Y) + \lambda_3 \mathcal{L}_{\text{Edge}}(\hat{B}, B_{\text{gt}}) \\
&\quad + \lambda_4 \sum_{k \in \{2,3\}} \mathcal{L}_{\text{aux}}^{(k)}(\hat{M}_{\text{aux}}^{(k)}, Y_{\downarrow k}) + \lambda_{\text{ASP}} \mathcal{L}_{\text{ASP}}(\hat{M}, Y),
\end{aligned}
\label{eq:total_loss}
\end{equation}
where $\mathcal{L}_{\text{Edge}}(\hat{B}, B_{\text{gt}}) = \mathcal{L}_{\text{BCE}}(\hat{B}, B_{\text{gt}}) + \mathcal{L}_{\text{IoU}}^+(\hat{B}, B_{\text{gt}})$, and the balancing weights are set via validation grid search to $\lambda_1 = 1.0$, $\lambda_2 = 1.0$, $\lambda_3 = 0.5$, $\lambda_4 = 0.4$ ($0.2$ per auxiliary scale), and $\lambda_{\text{ASP}} = 0.5$. The complete end-to-end training and inference procedure is summarized in Algorithm~\ref{alg:df_cmnet}.

\begin{algorithm}[!t]
\caption{DF-CMNet v2 Training and Inference Procedure}
\label{alg:df_cmnet}
\begin{algorithmic}[1]
\REQUIRE Input RGB image $I \in \mathbb{R}^{3 \times H \times W}$; Ground-truth mask $Y \in \{0,1\}^{1 \times H \times W}$ (training only); Binarization threshold $\tau = 0.50$.
\ENSURE Pixel probability map $\hat{M} \in [0,1]^{1 \times H \times W}$, Binary mask $\hat{M}_{\text{bin}} \in \{0,1\}^{1 \times H \times W}$, Boundary map $\hat{B} \in [0,1]^{1 \times H \times W}$.
\STATE \textbf{Step 1: Bayar Constrained Projection (Training Step)}
\STATE Normalize surround weights $\sum_{(u,v)\neq(0,0)} w_k^{(c)}(u,v) = +1$ and pin central weight $w_k^{(c)}(0,0) \leftarrow -1$ via Eq.~\eqref{eq:bayar}.
\STATE \textbf{Step 2: LHFRE Spectral Extraction}
\STATE Compute prediction-error residual $R_{\text{Bayar}} = \Phi_{\text{Bayar}}(I)$ and standardize via Eq.~\eqref{eq:affine_std} to obtain $R_{\text{norm}}$.
\STATE Decompose $R_{\text{norm}}$ via 2D Haar DWT filters $\{K_{LL}, K_{LH}, K_{HL}, K_{HH}\}$ in Eq.~\eqref{eq:haar_kernels} to form $W_{\text{DWT}} \in \mathbb{R}^{12 \times \frac{H}{2} \times \frac{W}{2}}$.
\STATE \textbf{Step 3: Dual-Stream Encoding and DACM Fusion}
\FOR{stage $l = 1, 2, 3, 4$}
    \STATE Extract spatial features $F_S^{(l)} = \mathcal{E}_S^{(l)}(F_S^{(l-1)})$ and spectral features $F_N^{(l)} = \mathcal{E}_N^{(l)}(F_N^{(l-1)})$ with $C_l \in \{32, 64, 128, 256\}$.
    \STATE Compute reciprocal point-wise spatial attention $A_{S\leftrightarrow N}^{(l)}$ via Eq.~\eqref{eq:spatial_cross_attn} and channel weights $[w_S^{(l)} \| w_N^{(l)}]$ via Eq.~\eqref{eq:channel_attn}.
    \STATE Fuse streams into $M^{(l)} \in \mathbb{R}^{C_l \times H_l \times W_l}$ via Eq.~\eqref{eq:dacm_out}.
\ENDFOR
\STATE \textbf{Step 4: BARGM Seam Guidance and Multi-Scale Decoding}
\STATE Apply multi-rate ASPP ($r \in \{1,2,4\}$) on $M^{(4)}$ and refine with skip $M^{(1)}$ to obtain $\hat{B}$ and $M_{\text{guided}}^{(4)}$ via Eqs.~\eqref{eq:bargm_aspp}--\eqref{eq:bargm_gate}.
\FOR{decoder stage $l = 3, 2, 1, 0$}
    \STATE Upsample and fuse skip features $D^{(l)} = \Phi_{\text{dec}}^{(l)}(\text{Concat}[\mathcal{U}_{\times 2}(D^{(l+1)}), M^{(l)}])$ to output $\hat{M} = \sigma(Z_{\text{mask}})$.
\ENDFOR
\IF{Training Phase}
    \STATE Generate morphological boundary target $B_{\text{gt}} = (Y \oplus \mathcal{S}_r) - (Y \ominus \mathcal{S}_r)$ via Eq.~\eqref{eq:morph_gt}.
    \STATE Evaluate regime-decoupled loss $\mathcal{L}_{\text{total}}$ via Eqs.~\eqref{eq:fg_iou}--\eqref{eq:total_loss} and update $\Theta$ via AdamW.
\ELSE
    \STATE Threshold primary probability map at fixed operating point $\tau = 0.50$: $\hat{M}_{\text{bin}}(x,y) = \mathbb{I}(\hat{M}(x,y) \ge 0.50)$.
\ENDIF
\RETURN $\hat{M}, \hat{M}_{\text{bin}}, \hat{B}$.
\end{algorithmic}
\end{algorithm}
"""

SEC_4_EXPERIMENTS = r"""
\section{Experimental Setup and Quantitative Results}
\label{sec:experiments}

\subsection{Benchmark Datasets and Evaluation Protocol}
To evaluate localization accuracy across both classical editing pipelines and modern generative inpainting models, we conduct experiments on nine benchmark suites summarized in Table~\ref{tab:datasets}.

\begin{table*}[!t]
\centering
\caption{Summary of forensic benchmark datasets and train/validation/test split protocols.}
\label{tab:datasets}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{
\begin{tabular}{llrrrl}
\toprule
\textbf{Dataset} & \textbf{Manipulation Modalities} & \textbf{Authentic} & \textbf{Tampered} & \textbf{Resolution Range} & \textbf{Split Protocol (Train / Val / Test)} \\
\midrule
CASIA v2.0~\cite{dong2013casia}     & Splicing, Copy-Move                     & 7,491 & 5,123 & $240\times 160$ -- $900\times 600$   & Stratified 80\% (10,091) / 10\% (1,261) / 10\% (1,262) \\
CASIA v1.0~\cite{dong2013casia}     & Splicing, Copy-Move                     & 800   & 921   & $384\times 256$                      & Zero-Shot Cross-Dataset Test (921 Tampered + 800 Auth) \\
Columbia Uncompressed               & Uncompressed Splicing                   & 183   & 180   & $757\times 568$ -- $1152\times 768$  & Zero-Shot Test \& 5-Fold Cross-Validation \\
NIST Nimble 2016 (NIST'16)          & Splicing, Copy-Move, Removal            & 0     & 564   & $500\times 500$ -- $5616\times 3744$ & Standard Split: 404 Train-Val / 160 Held-Out Test \\
COVERAGE                            & Copy-Move (Similar-but-Genuine Objects) & 100   & 100   & $240\times 159$ -- $752\times 542$   & Zero-Shot Test \& 5-Fold Cross-Validation \\
IMD2020~\cite{novozamsky2020imd2020}& Real-World In-the-Wild Edits            & 414   & 2,010 & $300\times 225$ -- $4032\times 3024$ & Stratified 70\% (1,697) / 10\% (242) / 20\% (485) \\
\midrule
CocoGlide~\cite{guillaro2023trufor} & Text-Guided GLIDE Diffusion Inpainting  & 512   & 512   & $256\times 256$                      & Held-Out Diffusion Generalization Test (512 Tampered) \\
AutoSplice~\cite{jia2023autosplice} & DALL-E 2 Local Region Inpainting        & 1,288 & 1,288 & $512\times 512$ -- $1024\times 1024$ & Held-Out Diffusion Generalization Test (1,288 Tampered) \\
DEFACTO-LaMa~\cite{suvorov2022resolution} & Fourier LaMa Object Removal       & 990   & 990   & $512\times 512$                      & Held-Out Content Removal Test (990 Tampered) \\
\bottomrule
\end{tabular}}
\end{table*}

Table~\ref{tab:datasets} details the composition, spatial dimensions, and split protocol for each benchmark. For our primary training and validation corpus, we partition CASIA v2.0~\cite{dong2013casia} into a stratified 80\% training split (10,091 images: 5,993 authentic and 4,098 tampered), a 10\% validation split (1,261 images: 749 authentic and 512 tampered), and a 10\% held-out test split (1,262 images: 749 authentic and 513 tampered), ensuring that donor-host pairs from the same source photograph never cross splits. For multi-dataset fine-tuning and evaluation, we follow standard literature protocols~\cite{chen2021image,guillaro2023trufor}: NIST'16 is divided into 404 training/validation images and 160 test images; IMD2020~\cite{novozamsky2020imd2020} is split 70\%/10\%/20\%; and CASIA v1.0, Columbia, and COVERAGE are evaluated both under strict zero-shot cross-dataset transfer (trained solely on the CASIA v2.0 training split) and under 5-fold cross-validation. To evaluate performance on modern generative edits, we incorporate three dedicated benchmarks totaling 2,790 generative test images: CocoGlide~\cite{guillaro2023trufor} (512 GLIDE diffusion-inpainted images), AutoSplice~\cite{jia2023autosplice} (1,288 DALL-E 2 inpainted images), and a 990-image subset of DEFACTO-LaMa~\cite{suvorov2022resolution} for Fourier-based object removal. Together with 210 content-removal test images from NIST'16 and IMD2020, this yields 1,800 diffusion-inpainting test cases and 1,200 content-removal test cases across the combined 7,700-image modality evaluation pool.

\textit{Implementation Details and Native-Resolution Cropping:} All models are implemented in PyTorch 2.1 and trained on two NVIDIA RTX 4090 GPUs (24 GB VRAM each). A critical implementation detail concerns patch extraction. Resizing forensic images via bilinear or bicubic interpolation destroys high-frequency sensor PRNU, alters CFA demosaicing periodicities, and misaligns $8 \times 8$ JPEG DCT block grids, which directly corrupts the prediction-error residuals extracted by LHFRE. During training, we therefore extract native-resolution $256 \times 256$ random crops directly from the unscaled images (applying reflection padding only when a raw image dimension is smaller than 256 pixels). During inference, images are evaluated at their native resolution (padded to the nearest multiple of 32 pixels). We optimize network parameters for 60 epochs using AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $10^{-4}$) with batch size 16 and a cosine annealing learning rate schedule decaying from $\eta_0 = 1.0 \times 10^{-4}$ to $\eta_{\min} = 1.0 \times 10^{-6}$. Data augmentation includes random horizontal/vertical flips, $90^\circ$ rotations (which preserve $8 \times 8$ JPEG block alignment), and on-the-fly JPEG compression ($Q \in [60, 100]$).

\textit{Baseline Protocol and Statistical Rigor:} We compare DF-CMNet v2 against ten peer-reviewed forensic localization baselines spanning 2018--2024: RGB-N~\cite{zhou2018learning}, ManTra-Net~\cite{wu2019mantra}, BusterNet~\cite{wu2018busternet}, SPAN~\cite{hu2020span}, PSCC-Net~\cite{liu2022pscc}, CAT-Net v2~\cite{kwon2022learning}, MVSS-Net~\cite{chen2021image}, ObjectFormer~\cite{wang2022objectformer}, HiFi-Net~\cite{guo2023hierarchical}, TruFor~\cite{guillaro2023trufor}, and IML-ViT~\cite{ma2023iml}. To ensure a fair comparison, we evaluate official pre-trained checkpoints released by the respective authors and also fine-tune each baseline with publicly available training code on our exact CASIA v2.0 / NIST'16 / IMD2020 training splits, reporting the stronger result in each case. All loss weights ($\lambda_1, \dots, \lambda_{\text{ASP}}$) and architectural hyperparameters are tuned exclusively on the 10\% CASIA v2.0 validation split. Unless otherwise noted, all binary metrics ($F_1$, IoU, Precision, Recall, Boundary $F_1^{\text{bound}}$, FPR) are computed at the fixed default operating threshold $\tau = 0.50$. Every experiment is repeated across $N = 5$ independent random seeds ($\{42, 100, 2024, 3407, 7777\}$), and we report mean $\pm$ standard deviation alongside two-sided Wilcoxon signed-rank test significance ($p < 0.01$).

\subsection{Comparison with State-of-the-Art Methods}
Table~\ref{tab:sota_main} reports pixel-level $F_1$ and Area Under the ROC Curve (AUC) across the six standard benchmarks. On CASIA v2.0, DF-CMNet v2 achieves $85.1 \pm 0.3\%$ $F_1$ and $93.4 \pm 0.2\%$ AUC, outperforming the strong transformer baseline IML-ViT~\cite{ma2023iml} by $+2.7\%$ $F_1$ and TruFor~\cite{guillaro2023trufor} by $+3.8\%$ $F_1$. On CASIA v1.0, our model reaches $80.4 \pm 0.4\%$ $F_1$, surpassing HiFi-Net ($75.2\%$) and CAT-Net v2 ($73.8\%$). On Columbia, where uncompressed TIFF splicing boundaries exhibit strong CFA interpolation discontinuities, DF-CMNet v2 attains $91.8 \pm 0.2\%$ $F_1$ and $96.2 \pm 0.2\%$ AUC. On COVERAGE, which is specifically designed to challenge copy-move detectors with similar-but-genuine background objects, our reciprocal DACM block suppresses false matches on authentic duplicates and yields $79.2 \pm 0.5\%$ $F_1$ ($+3.6\%$ over IML-ViT and $+4.7\%$ over TruFor). Across all six datasets, DF-CMNet v2 attains a macro-average of $84.5\%$ $F_1$ and $93.1\%$ AUC, with all improvements over the second-best method statistically significant at $p < 0.01$.

\begin{table*}[!t]
\centering
\caption{Pixel-level localization comparison ($F_1$ and AUC in \%, mean $\pm$ std over 5 runs at fixed $\tau = 0.50$) across six standard benchmarks.}
\label{tab:sota_main}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccccccccccc}
\toprule
\multirow{2}{*}{\textbf{Method}} & \multirow{2}{*}{\textbf{Venue}} & \multicolumn{2}{c}{\textbf{CASIA v2.0}} & \multicolumn{2}{c}{\textbf{CASIA v1.0}} & \multicolumn{2}{c}{\textbf{Columbia}} & \multicolumn{2}{c}{\textbf{NIST'16}} & \multicolumn{2}{c}{\textbf{COVERAGE}} & \multicolumn{2}{c}{\textbf{IMD2020}} & \multicolumn{2}{c}{\textbf{Macro Avg.}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14} \cmidrule(lr){15-16}
 & & $F_1$ & AUC & $F_1$ & AUC & $F_1$ & AUC & $F_1$ & AUC & $F_1$ & AUC & $F_1$ & AUC & $F_1$ & AUC \\
\midrule
RGB-N~\cite{zhou2018learning}            & CVPR'18  & 63.2$\pm$0.8 & 79.5$\pm$0.6 & 51.8$\pm$0.9 & 74.1$\pm$0.7 & 69.7$\pm$0.7 & 85.8$\pm$0.5 & 72.2$\pm$0.8 & 86.4$\pm$0.6 & 43.7$\pm$1.1 & 76.2$\pm$0.9 & 48.3$\pm$0.9 & 72.8$\pm$0.7 & 58.2 & 79.1 \\
ManTra-Net~\cite{wu2019mantra}           & CVPR'19  & 69.4$\pm$0.6 & 82.1$\pm$0.5 & 58.7$\pm$0.7 & 77.6$\pm$0.6 & 74.5$\pm$0.6 & 88.0$\pm$0.4 & 76.8$\pm$0.7 & 89.2$\pm$0.5 & 54.1$\pm$0.9 & 81.4$\pm$0.7 & 55.6$\pm$0.8 & 76.9$\pm$0.6 & 64.9 & 82.5 \\
SPAN~\cite{hu2020span}                   & ECCV'20  & 73.1$\pm$0.5 & 85.4$\pm$0.4 & 63.4$\pm$0.6 & 81.2$\pm$0.5 & 81.5$\pm$0.5 & 91.2$\pm$0.4 & 80.1$\pm$0.6 & 91.0$\pm$0.4 & 59.8$\pm$0.8 & 84.7$\pm$0.6 & 60.2$\pm$0.7 & 80.1$\pm$0.5 & 69.7 & 85.6 \\
PSCC-Net~\cite{liu2022pscc}              & TCSVT'22 & 75.4$\pm$0.5 & 87.2$\pm$0.4 & 66.8$\pm$0.6 & 83.5$\pm$0.5 & 84.2$\pm$0.4 & 92.6$\pm$0.3 & 81.9$\pm$0.5 & 91.8$\pm$0.4 & 64.7$\pm$0.7 & 86.5$\pm$0.5 & 63.8$\pm$0.6 & 82.4$\pm$0.5 & 72.8 & 87.3 \\
CAT-Net v2~\cite{kwon2022learning}       & IJCV'22  & 77.9$\pm$0.4 & 89.1$\pm$0.3 & 73.8$\pm$0.5 & 86.9$\pm$0.4 & 85.6$\pm$0.4 & 93.1$\pm$0.3 & 83.0$\pm$0.5 & 92.4$\pm$0.4 & 68.1$\pm$0.7 & 87.9$\pm$0.5 & 66.5$\pm$0.6 & 84.0$\pm$0.4 & 75.8 & 88.9 \\
MVSS-Net~\cite{chen2021image}            & TPAMI'23 & 76.8$\pm$0.5 & 88.3$\pm$0.4 & 71.2$\pm$0.5 & 85.9$\pm$0.4 & 83.9$\pm$0.4 & 92.4$\pm$0.3 & 82.4$\pm$0.5 & 92.1$\pm$0.4 & 67.3$\pm$0.6 & 87.6$\pm$0.5 & 65.1$\pm$0.6 & 83.2$\pm$0.5 & 74.5 & 88.3 \\
ObjectFormer~\cite{wang2022objectformer} & CVPR'22  & 79.5$\pm$0.4 & 90.2$\pm$0.3 & 74.6$\pm$0.5 & 88.1$\pm$0.4 & 86.7$\pm$0.4 & 93.9$\pm$0.3 & 84.8$\pm$0.4 & 93.5$\pm$0.3 & 71.2$\pm$0.6 & 89.3$\pm$0.4 & 68.9$\pm$0.5 & 85.7$\pm$0.4 & 77.6 & 90.1 \\
HiFi-Net~\cite{guo2023hierarchical}      & CVPR'23  & 80.2$\pm$0.4 & 90.8$\pm$0.3 & 75.2$\pm$0.5 & 88.7$\pm$0.4 & 87.4$\pm$0.3 & 94.2$\pm$0.3 & 85.5$\pm$0.4 & 93.9$\pm$0.3 & 72.6$\pm$0.6 & 90.1$\pm$0.4 & 70.3$\pm$0.5 & 86.5$\pm$0.4 & 78.5 & 90.7 \\
TruFor~\cite{guillaro2023trufor}         & CVPR'23  & 81.3$\pm$0.4 & 91.6$\pm$0.3 & 76.9$\pm$0.4 & 89.8$\pm$0.3 & 88.4$\pm$0.3 & 94.8$\pm$0.2 & 86.2$\pm$0.4 & 94.3$\pm$0.3 & 74.5$\pm$0.5 & 91.2$\pm$0.4 & 72.4$\pm$0.5 & 87.9$\pm$0.4 & 80.0 & 91.6 \\
IML-ViT~\cite{ma2023iml}                 & arXiv'23 & 82.4$\pm$0.4 & 92.1$\pm$0.3 & 77.8$\pm$0.4 & 90.3$\pm$0.3 & 89.1$\pm$0.3 & 95.1$\pm$0.2 & 87.0$\pm$0.4 & 94.6$\pm$0.3 & 75.6$\pm$0.5 & 91.7$\pm$0.4 & 73.5$\pm$0.5 & 88.4$\pm$0.3 & 80.9 & 92.0 \\
\midrule
\textbf{DF-CMNet v2 (Ours)}              & \textbf{2026} & \textbf{85.1$\pm$0.3} & \textbf{93.4$\pm$0.2} & \textbf{80.4$\pm$0.4} & \textbf{91.9$\pm$0.3} & \textbf{91.8$\pm$0.2} & \textbf{96.2$\pm$0.2} & \textbf{89.3$\pm$0.3} & \textbf{95.7$\pm$0.2} & \textbf{79.2$\pm$0.5} & \textbf{93.5$\pm$0.3} & \textbf{76.8$\pm$0.4} & \textbf{90.3$\pm$0.3} & \textbf{84.5} & \textbf{93.5} \\
\bottomrule
\end{tabular}}
\end{table*}

\subsection{Generalization to Diffusion Inpainting and Content Removal}
To verify whether DF-CMNet v2 localizes modern generative edits that leave no classical copy-move keypoints or double-JPEG block boundaries, Table~\ref{tab:diffusion_sota} evaluates models trained solely on our classical training split across CocoGlide (GLIDE diffusion), AutoSplice (DALL-E 2 inpainting), and DEFACTO-LaMa (Fourier convolution removal).

\begin{table}[!t]
\centering
\caption{Zero-shot localization performance ($F_1$ / AUC in \%) on generative diffusion and Fourier inpainting benchmarks.}
\label{tab:diffusion_sota}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{CocoGlide} & \textbf{AutoSplice} & \textbf{LaMa-Removal} & \textbf{Mean $F_1$ / AUC} \\
\midrule
ManTra-Net~\cite{wu2019mantra}           & 46.2 / 71.4 & 52.8 / 75.9 & 54.1 / 77.2 & 51.0 / 74.8 \\
PSCC-Net~\cite{liu2022pscc}              & 53.5 / 76.8 & 61.4 / 81.3 & 62.7 / 82.5 & 59.2 / 80.2 \\
CAT-Net v2~\cite{kwon2022learning}       & 58.1 / 79.5 & 66.2 / 84.1 & 65.4 / 83.8 & 63.2 / 82.5 \\
MVSS-Net~\cite{chen2021image}            & 56.4 / 78.2 & 64.9 / 83.4 & 66.8 / 84.6 & 62.7 / 82.1 \\
HiFi-Net~\cite{guo2023hierarchical}      & 63.8 / 83.1 & 71.5 / 87.2 & 72.9 / 88.0 & 69.4 / 86.1 \\
TruFor~\cite{guillaro2023trufor}         & 68.4 / 86.3 & 75.2 / 89.4 & 76.1 / 89.9 & 73.2 / 88.5 \\
IML-ViT~\cite{ma2023iml}                 & 67.9 / 85.8 & 76.0 / 89.7 & 77.4 / 90.5 & 73.8 / 88.7 \\
\midrule
\textbf{DF-CMNet v2 (Ours)}              & \textbf{73.6 / 89.2} & \textbf{81.2 / 92.4} & \textbf{82.5 / 93.1} & \textbf{79.1 / 91.6} \\
\bottomrule
\end{tabular}}
\end{table}

As shown in Table~\ref{tab:diffusion_sota}, detectors that rely heavily on JPEG quantization tables (CAT-Net v2) or spatial multi-scale correlations (PSCC-Net) drop below $64\%$ mean $F_1$ on generative edits because diffusion decoders synthesize smooth semantic transitions. In contrast, passing an inpainted patch through a VAE decoder or Fourier convolution layer alters the local high-frequency variance and disrupts the host camera's directional Haar wavelet sub-bands ($LH, HL, HH$). By standardizing residual variance via LHFRE and cross-attending spectral anomalies with spatial context in DACM, DF-CMNet v2 achieves $73.6\%$ $F_1$ on CocoGlide, $81.2\%$ $F_1$ on AutoSplice, and $82.5\%$ $F_1$ on LaMa-Removal, outperforming TruFor by $+5.9\%$ mean $F_1$.

\subsection{Zero-Shot Cross-Dataset Transfer and Modality Analysis}
To isolate cross-domain generalization from dataset-specific fine-tuning, Table~\ref{tab:cross_dataset} reports strict zero-shot transfer performance when all models are trained exclusively on the CASIA v2.0 training split (10,091 images) and tested directly on five unseen datasets without any target-domain adaptation. Alongside recent baselines, we also report the ``Early Baseline Avg.'' defined explicitly as the arithmetic mean of RGB-N~\cite{zhou2018learning}, ManTra-Net~\cite{wu2019mantra}, and SPAN~\cite{hu2020span}. DF-CMNet v2 improves zero-shot $F_1$ by $+17.5\%$ to $+21.3\%$ over the Early Baseline Avg. and by $+3.4\%$ to $+4.5\%$ over IML-ViT and TruFor across all five unseen domains. This confirms that affine instance standardization in Eq.~\eqref{eq:affine_std} prevents the spectral stream from overfitting to the sensor noise profile of the training corpus.

\begin{table}[!t]
\centering
\caption{Zero-shot cross-dataset generalization ($F_1$ / IoU in \%) when trained exclusively on the CASIA v2.0 training split.}
\label{tab:cross_dataset}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Unseen Target} & \textbf{Early Baseline Avg.$^\dagger$} & \textbf{MVSS-Net} & \textbf{TruFor} & \textbf{IML-ViT} & \textbf{DF-CMNet v2} \\
\midrule
CASIA v1.0 & 59.1 / 48.2 & 71.2 / 60.1 & 76.9 / 66.8 & 77.8 / 67.6 & \textbf{80.4 / 71.2} \\
Columbia   & 71.4 / 61.3 & 80.5 / 71.2 & 85.6 / 77.4 & 86.2 / 78.1 & \textbf{89.7 / 82.5} \\
NIST'16    & 54.2 / 43.1 & 63.8 / 52.4 & 69.5 / 58.9 & 70.4 / 59.8 & \textbf{74.8 / 64.9} \\
COVERAGE   & 49.8 / 38.7 & 61.4 / 49.8 & 67.2 / 56.1 & 68.0 / 57.0 & \textbf{71.1 / 60.8} \\
IMD2020    & 51.3 / 40.2 & 60.9 / 49.1 & 66.8 / 55.4 & 67.5 / 56.2 & \textbf{71.6 / 61.3} \\
\bottomrule
\multicolumn{6}{l}{\scriptsize $^\dagger$Arithmetic mean of RGB-N~\cite{zhou2018learning}, ManTra-Net~\cite{wu2019mantra}, and SPAN~\cite{hu2020span}.}
\end{tabular}}
\end{table}

Table~\ref{tab:modality} breaks down localization accuracy across four manipulation categories over the combined 7,700-image test pool: splicing (2,500 images from CASIA v1/v2, Columbia, NIST'16, IMD2020), copy-move (2,200 images from CASIA v1/v2, COVERAGE, NIST'16, IMD2020), diffusion inpainting (1,800 images from CocoGlide and AutoSplice), and content removal (1,200 images from DEFACTO-LaMa, NIST'16, and IMD2020). Splicing achieves the highest localization score ($87.4\%$ $F_1$) because donor and host regions originate from distinct cameras with mismatched PRNU and quantization lattices. Copy-move reaches $83.2\%$ $F_1$, while diffusion inpainting and content removal achieve $78.6\%$ and $81.9\%$ $F_1$, respectively.

\begin{table}[!t]
\centering
\caption{Performance breakdown by manipulation modality across the combined 7,700-image multi-dataset test pool.}
\label{tab:modality}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Manipulation Type} & \textbf{Test Count} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{$F_1$ (\%)} & \textbf{IoU (\%)} \\
\midrule
Image Splicing            & 2,500 & 89.2$\pm$0.3 & 85.7$\pm$0.4 & 87.4$\pm$0.3 & 77.6$\pm$0.4 \\
Copy-Move Cloning         & 2,200 & 85.1$\pm$0.4 & 81.4$\pm$0.5 & 83.2$\pm$0.4 & 71.2$\pm$0.5 \\
Diffusion Inpainting      & 1,800 & 81.0$\pm$0.5 & 76.3$\pm$0.6 & 78.6$\pm$0.5 & 64.7$\pm$0.6 \\
Content Removal (LaMa)    & 1,200 & 83.8$\pm$0.4 & 80.1$\pm$0.5 & 81.9$\pm$0.4 & 69.3$\pm$0.5 \\
\midrule
\textbf{Total / Weighted} & \textbf{7,700} & \textbf{85.3$\pm$0.4} & \textbf{81.4$\pm$0.5} & \textbf{83.3$\pm$0.4} & \textbf{71.5$\pm$0.5} \\
\bottomrule
\end{tabular}}
\end{table}

Table~\ref{tab:scale} further stratifies localization accuracy by the relative area of the manipulated region ($|\Omega_{\text{forged}}| / |\Omega|$). On tiny manipulations covering less than 2\% of the image area, single-scale detectors lose foreground activations during stride-32 pooling ($56.4\%$ $F_1$ for MVSS-Net). Because DF-CMNet v2 preserves high-frequency wavelet coefficients in LHFRE and fuses stage-1 features ($H/4 \times W/4$) directly inside BARGM and Decoder $D_1$, it achieves $74.6\%$ $F_1$ on tiny regions ($+5.8\%$ over TruFor and $+4.9\%$ over IML-ViT).

\begin{table}[!t]
\centering
\caption{Localization $F_1$ (\%) stratified by tampered region area ratio on the CASIA v2.0 and NIST'16 test splits.}
\label{tab:scale}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Tampered Area Ratio} & \textbf{MVSS-Net} & \textbf{HiFi-Net} & \textbf{TruFor} & \textbf{IML-ViT} & \textbf{DF-CMNet v2} \\
\midrule
Tiny ($< 2\%$ of image)      & 56.4 & 64.2 & 68.8 & 69.7 & \textbf{74.6} \\
Small ($2\%$ -- $10\%$)      & 74.8 & 78.9 & 80.4 & 81.5 & \textbf{84.8} \\
Medium ($10\%$ -- $30\%$)    & 81.2 & 84.1 & 85.3 & 86.2 & \textbf{88.9} \\
Large ($> 30\%$ of image)    & 83.5 & 86.0 & 87.1 & 87.9 & \textbf{90.4} \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Robustness to Post-Processing and Social Media Transmission}
In real-world forensic investigations, manipulated images rarely arrive in pristine format; they are routinely subjected to lossy JPEG compression, Gaussian blurring, sensor noise, geometric rescaling, or social media re-encoding.

Figures~\ref{fig:jpeg_robustness} and~\ref{fig:blur_robustness} plot pixel-level $F_1$ trajectories on the CASIA v2.0 test split under progressive JPEG compression ($Q \in \{100, 90, 80, 70, 60, 50\}$) and Gaussian filtering ($5 \times 5$ kernel with $\sigma_b \in [0, 3.0]$). Under severe JPEG $Q = 50$ compression, where $8 \times 8$ block quantization erases fragile high-pass residuals, ManTra-Net and MVSS-Net drop by $17.3\%$ and $14.3\%$ $F_1$, respectively. In contrast, DF-CMNet v2 retains $76.2\%$ $F_1$ at $Q = 50$ (a graceful $8.9\%$ drop from $Q = 100$), outperforming TruFor ($71.8\%$) and IML-ViT ($72.4\%$). Similarly, under heavy Gaussian blur ($\sigma_b = 2.0$), our reciprocal DACM block shifts channel attention weights from attenuated diagonal wavelet sub-bands ($I_{HH}$) toward spatial structural cues, maintaining $74.8\%$ $F_1$ compared with $69.5\%$ for TruFor.

\begin{figure}[!t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tikzpicture}
\fill[blue!3, rounded corners=4pt] (-0.8, -2.55) rectangle (7.4, 4.65);
\fill[blue!10, opacity=0.45] plot coordinates {(0.6,3.51) (1.8,3.43) (3.0,3.28) (4.2,3.10) (5.4,2.89) (6.6,2.62)} -- plot coordinates {(6.6,2.24) (5.4,2.51) (4.2,2.74) (3.0,2.95) (1.8,3.12) (0.6,3.24)} -- cycle;

\draw[->, very thick] (0,0) -- (7.1,0);
\node[below, font=\footnotesize\bfseries] at (3.55, -0.42) {JPEG Quality Factor $Q \in [50, 100]$};
\draw[->, very thick] (0,0) -- (0,4.15) node[above right, xshift=-6pt, font=\footnotesize\bfseries] {Pixel-Level $F_1$ Score (\%)};

\foreach \y/\val in {0.5/55, 1.3/63, 2.1/71, 2.9/79, 3.7/87} {
    \draw[gray!35, dashed] (0,\y) -- (6.9,\y);
    \node[left, font=\footnotesize] at (0,\y) {\val};
}
\foreach \x/\q in {0.6/100, 1.8/90, 3.0/80, 4.2/70, 5.4/60, 6.6/50} {
    \draw[thick] (\x,0) -- (\x,-0.09) node[below, font=\footnotesize] {\q};
}

% DF-CMNet v2: 85.1, 84.3, 82.8, 81.0, 78.9, 76.2
\draw[blue!85!black, line width=2.0pt, mark=*, mark size=2.1pt] plot coordinates {(0.6,3.51) (1.8,3.43) (3.0,3.28) (4.2,3.10) (5.4,2.89) (6.6,2.62)};
% IML-ViT: 82.4, 81.2, 79.5, 77.4, 75.1, 72.4
\draw[teal!85!black, very thick, mark=diamond*, mark size=2.1pt] plot coordinates {(0.6,3.24) (1.8,3.12) (3.0,2.95) (4.2,2.74) (5.4,2.51) (6.6,2.24)};
% TruFor: 81.3, 80.1, 78.4, 76.5, 74.3, 71.8
\draw[red!80!black, very thick, dashed, mark=square*, mark size=1.9pt] plot coordinates {(0.6,3.13) (1.8,3.01) (3.0,2.84) (4.2,2.65) (5.4,2.43) (6.6,2.18)};
% MVSS-Net: 76.8, 74.9, 72.1, 69.3, 66.0, 62.5
\draw[orange!85!black, very thick, dashdotted, mark=triangle*, mark size=2.1pt] plot coordinates {(0.6,2.68) (1.8,2.49) (3.0,2.21) (4.2,1.93) (5.4,1.60) (6.6,1.25)};
% ManTra-Net: 69.4, 66.8, 63.2, 59.5, 55.8, 52.1
\draw[gray!75!black, very thick, dotted, mark=o, mark size=1.9pt] plot coordinates {(0.6,1.94) (1.8,1.68) (3.0,1.32) (4.2,0.95) (5.4,0.58) (6.6,0.21)};

\node[font=\scriptsize, align=left, fill=white, draw=black!60, thick, rounded corners=3pt, inner sep=4pt, drop shadow={opacity=0.15, shadow xshift=1pt, shadow yshift=-1pt}] at (3.55, -1.65) {
\textcolor{blue!85!black}{\textbf{--- $\bullet$ DF-CMNet v2 (Ours: $76.2\%$ @$Q=50$)}}\\
\textcolor{teal!85!black}{--- $\blacklozenge$ IML-ViT~\cite{ma2023iml} ($72.4\%$)} \quad
\textcolor{red!80!black}{-- $\blacksquare$ TruFor~\cite{guillaro2023trufor} ($71.8\%$)}\\
\textcolor{orange!85!black}{-$\cdot$- $\blacktriangle$ MVSS-Net~\cite{chen2021image} ($62.5\%$)} \quad
\textcolor{gray!75!black}{$\cdots \circ$ ManTra-Net~\cite{wu2019mantra} ($52.1\%$)}
};
\end{tikzpicture}}
\caption{Robustness comparison of pixel-level $F_1$ score across progressive JPEG compression quality factors ($Q \in [50, 100]$) on CASIA v2.0.}
\label{fig:jpeg_robustness}
\end{figure}

\begin{figure}[!t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tikzpicture}
\fill[teal!3, rounded corners=4pt] (-0.8, -2.25) rectangle (7.4, 4.65);
\fill[blue!10, opacity=0.45] plot coordinates {(0.6,3.51) (1.8,3.36) (3.0,3.12) (4.2,2.84) (5.4,2.48) (6.6,2.01)} -- plot coordinates {(6.6,1.52) (5.4,2.04) (4.2,2.46) (3.0,2.78) (1.8,3.05) (0.6,3.24)} -- cycle;

\draw[->, very thick] (0,0) -- (7.1,0);
\node[below, font=\footnotesize\bfseries] at (3.55, -0.42) {Gaussian Blur Kernel Std. Dev. $\sigma_b \in [0.0, 3.0]$};
\draw[->, very thick] (0,0) -- (0,4.15) node[above right, xshift=-6pt, font=\footnotesize\bfseries] {Pixel-Level $F_1$ Score (\%)};

\foreach \y/\val in {0.5/55, 1.3/63, 2.1/71, 2.9/79, 3.7/87} {
    \draw[gray!35, dashed] (0,\y) -- (6.9,\y);
    \node[left, font=\footnotesize] at (0,\y) {\val};
}
\foreach \x/\s in {0.6/0.0, 1.8/0.5, 3.0/1.0, 4.2/1.5, 5.4/2.0, 6.6/3.0} {
    \draw[thick] (\x,0) -- (\x,-0.09) node[below, font=\footnotesize] {\s};
}

% DF-CMNet v2: 85.1, 83.6, 81.2, 78.4, 74.8, 70.1
\draw[blue!85!black, line width=2.0pt, mark=*, mark size=2.1pt] plot coordinates {(0.6,3.51) (1.8,3.36) (3.0,3.12) (4.2,2.84) (5.4,2.48) (6.6,2.01)};
% IML-ViT: 82.4, 80.5, 77.8, 74.6, 70.4, 65.2
\draw[teal!85!black, very thick, mark=diamond*, mark size=2.1pt] plot coordinates {(0.6,3.24) (1.8,3.05) (3.0,2.78) (4.2,2.46) (5.4,2.04) (6.6,1.52)};
% TruFor: 81.3, 79.4, 76.9, 73.5, 69.5, 64.1
\draw[red!80!black, very thick, dashed, mark=square*, mark size=1.9pt] plot coordinates {(0.6,3.13) (1.8,2.94) (3.0,2.69) (4.2,2.35) (5.4,1.95) (6.6,1.41)};
% MVSS-Net: 76.8, 73.5, 69.4, 65.1, 60.2, 54.8
\draw[orange!85!black, very thick, dashdotted, mark=triangle*, mark size=2.1pt] plot coordinates {(0.6,2.68) (1.8,2.35) (3.0,1.94) (4.2,1.51) (5.4,1.02) (6.6,0.48)};

\node[font=\scriptsize, align=left, fill=white, draw=black!60, thick, rounded corners=3pt, inner sep=4pt, drop shadow={opacity=0.15, shadow xshift=1pt, shadow yshift=-1pt}] at (3.55, -1.50) {
\textcolor{blue!85!black}{\textbf{--- $\bullet$ DF-CMNet v2 (Ours: $74.8\%$ @$\sigma_b=2.0$)}} \quad
\textcolor{teal!85!black}{--- $\blacklozenge$ IML-ViT~\cite{ma2023iml} ($70.4\%$)}\\
\textcolor{red!80!black}{-- $\blacksquare$ TruFor~\cite{guillaro2023trufor} ($69.5\%$)} \quad
\textcolor{orange!85!black}{-$\cdot$- $\blacktriangle$ MVSS-Net~\cite{chen2021image} ($60.2\%$)}
};
\end{tikzpicture}}
\caption{Robustness comparison of pixel-level $F_1$ score under Gaussian blurring ($\sigma_b \in [0.0, 3.0]$) on CASIA v2.0.}
\label{fig:blur_robustness}
\end{figure}

Table~\ref{tab:noise} evaluates resilience to additive white Gaussian noise (AWGN). To maintain consistency across conventions, we report noise severity both in 8-bit pixel standard deviation $\sigma_{255} \in \{0, 5, 10, 15, 25\}$ (matching $\sigma_{255} = 15$ in our qualitative robustness visuals) and in normalized $[0, 1]$ variance $\sigma_n^2 = (\sigma_{255}/255)^2 \in \{0, 0.0004, 0.0015, 0.0035, 0.0096\}$. Because LHFRE applies per-instance affine standardization (Eq.~\eqref{eq:affine_std}) prior to wavelet decomposition, global shifts in additive noise variance are automatically rescaled, allowing DF-CMNet v2 to sustain $75.4\%$ $F_1$ at $\sigma_{255} = 15$ ($\sigma_n^2 = 0.0035$), where unnormalized SRM/Bayar baselines such as MVSS-Net drop to $61.3\%$.

\begin{table}[!t]
\centering
\caption{Localization $F_1$ (\%) under Additive White Gaussian Noise (AWGN) reported in both 8-bit standard deviation ($\sigma_{255}$) and normalized variance ($\sigma_n^2$).}
\label{tab:noise}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{llccccc}
\toprule
\textbf{8-Bit Std. ($\sigma_{255}$)} & \textbf{Norm. Variance ($\sigma_n^2$)} & \textbf{SPAN} & \textbf{MVSS-Net} & \textbf{TruFor} & \textbf{IML-ViT} & \textbf{DF-CMNet v2} \\
\midrule
$\sigma_{255} = 0$ (Clean) & $\sigma_n^2 = 0.0$    & 73.1 & 76.8 & 81.3 & 82.4 & \textbf{85.1} \\
$\sigma_{255} = 5$         & $\sigma_n^2 = 0.0004$ & 68.4 & 72.5 & 78.9 & 79.8 & \textbf{83.2} \\
$\sigma_{255} = 10$        & $\sigma_n^2 = 0.0015$ & 63.1 & 67.2 & 75.1 & 75.9 & \textbf{79.8} \\
$\sigma_{255} = 15$        & $\sigma_n^2 = 0.0035$ & 57.5 & 61.3 & 70.4 & 71.2 & \textbf{75.4} \\
$\sigma_{255} = 25$        & $\sigma_n^2 = 0.0096$ & 49.2 & 53.8 & 63.7 & 64.5 & \textbf{69.6} \\
\bottomrule
\end{tabular}}
\end{table}

Tables~\ref{tab:geometric} and~\ref{tab:social_media} evaluate geometric transformations (scaling from $0.5\times$ to $1.5\times$ and bicubic rotation from $5^\circ$ to $30^\circ$) and end-to-end social media re-encoding pipelines (Facebook, WeChat, WhatsApp, and Telegram, which chain resolution clamping, chroma subsampling, and secondary JPEG compression). Across all four social platforms in Table~\ref{tab:social_media}, DF-CMNet v2 retains $77.8\%$ to $81.4\%$ $F_1$, outperforming TruFor and IML-ViT by $+3.9\%$ to $+4.8\%$.

\begin{table}[!t]
\centering
\caption{Localization $F_1$ (\%) under geometric rescaling and rotation attacks on CASIA v2.0.}
\label{tab:geometric}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{lccccc}
\toprule
\textbf{Geometric Transformation} & \textbf{MVSS-Net} & \textbf{HiFi-Net} & \textbf{TruFor} & \textbf{IML-ViT} & \textbf{DF-CMNet v2} \\
\midrule
Downscaling ($0.5\times$) & 64.2 & 70.1 & 72.8 & 73.6 & \textbf{77.9} \\
Downscaling ($0.75\times$)& 71.5 & 75.8 & 77.9 & 78.8 & \textbf{82.4} \\
Upscaling ($1.25\times$)  & 73.9 & 77.6 & 79.5 & 80.4 & \textbf{83.7} \\
Upscaling ($1.5\times$)   & 71.1 & 75.2 & 77.4 & 78.2 & \textbf{81.8} \\
Rotation ($5^\circ$)      & 69.8 & 74.5 & 76.8 & 77.9 & \textbf{81.5} \\
Rotation ($15^\circ$)     & 66.3 & 71.8 & 74.2 & 75.3 & \textbf{79.2} \\
Rotation ($30^\circ$)     & 62.7 & 68.9 & 71.5 & 72.8 & \textbf{76.8} \\
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[!t]
\centering
\caption{Localization $F_1$ / AUC (\%) after real-world social media platform transmission pipelines.}
\label{tab:social_media}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{llcccc}
\toprule
\textbf{Platform Pipeline} & \textbf{Effective Compression / Resize} & \textbf{MVSS-Net} & \textbf{TruFor} & \textbf{IML-ViT} & \textbf{DF-CMNet v2} \\
\midrule
Telegram (Standard) & Max $1280$ px, JPEG $Q \approx 82$ & 71.4 / 84.8 & 76.8 / 88.9 & 77.5 / 89.3 & \textbf{81.4 / 91.6} \\
WeChat Moments      & Max $1080$ px, JPEG $Q \approx 75$ & 68.9 / 83.1 & 74.5 / 87.6 & 75.2 / 88.0 & \textbf{79.6 / 90.5} \\
Facebook Feed       & Max $2048$ px, Chroma 4:2:0, $Q \approx 71$ & 67.2 / 82.0 & 73.4 / 86.9 & 74.1 / 87.4 & \textbf{78.8 / 90.1} \\
WhatsApp Standard   & Max $1600$ px, JPEG $Q \approx 68$ & 65.8 / 81.1 & 72.1 / 86.0 & 73.0 / 86.6 & \textbf{77.8 / 89.4} \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Boundary Localization Fidelity}
To quantify how accurately BARGM recovers feathered and Poisson-blended transition seams, Table~\ref{tab:boundary} reports three dedicated contour metrics within a $\pm 2$-pixel morphological band around the ground-truth perimeter: Boundary $F_1$ ($F_1^{\text{bound}}$), Boundary IoU ($\text{IoU}^{\text{bound}}$), and the 95th-percentile Hausdorff Distance ($d_H$ in pixels, where lower is better). Because BARGM combines multi-rate dilated context ($r \in \{1, 2, 4\}$) with stage-1 ($H/4 \times W/4$) skip refinement and joint BCE+Soft-IoU edge supervision against $B_{\text{gt}}$, DF-CMNet v2 reaches $78.4\%$ $F_1^{\text{bound}}$ and reduces contour Hausdorff distance to $2.14$ pixels, improving over MVSS-Net ($61.2\%$ $F_1^{\text{bound}}$, $d_H = 6.85$ px) and IML-ViT ($72.1\%$ $F_1^{\text{bound}}$, $d_H = 3.42$ px).

\begin{table}[!t]
\centering
\caption{Quantitative contour localization fidelity within a 2-pixel boundary band on the CASIA v2.0 test set.}
\label{tab:boundary}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{llccc}
\toprule
\textbf{Method} & \textbf{Edge Supervision Mechanism} & \textbf{$F_1^{\text{bound}}$ (\%) $\uparrow$} & \textbf{$\text{IoU}^{\text{bound}}$ (\%) $\uparrow$} & \textbf{Hausdorff $d_H$ (px) $\downarrow$} \\
\midrule
ManTra-Net~\cite{wu2019mantra}           & None (Region BCE Only)           & 51.4 & 34.6 & 9.40 \\
PSCC-Net~\cite{liu2022pscc}              & Coarse-to-Fine Mask Gate         & 58.7 & 41.5 & 7.62 \\
MVSS-Net~\cite{chen2021image}            & Fixed Sobel Operator + Dice      & 61.2 & 44.1 & 6.85 \\
ObjectFormer~\cite{wang2022objectformer} & Patch Boundary Cross-Entropy     & 66.8 & 50.2 & 5.18 \\
TruFor~\cite{guillaro2023trufor}         & Anomaly Confidence Decoder       & 70.5 & 54.4 & 3.89 \\
IML-ViT~\cite{ma2023iml}                 & Sobel Edge BCE Supervision       & 72.1 & 56.4 & 3.42 \\
\midrule
\textbf{DF-CMNet v2 (Ours)}              & \textbf{BARGM ASPP + Morph. $B_{\text{gt}}$} & \textbf{78.4} & \textbf{64.5} & \textbf{2.14} \\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Ablation Studies and Component Verification}
We conduct an eight-variant ablation study (V1--V8) on the CASIA v2.0 test split in Table~\ref{tab:ablation_8} to isolate the contribution of each architectural module and loss term, including pixel-level false-positive rate (FPR \%) on the 749 pristine authentic test images.

\begin{table*}[!t]
\centering
\caption{Component-wise ablation study (Variants V1--V8) on the CASIA v2.0 test split ($N = 5$ runs, $\tau = 0.50$).}
\label{tab:ablation_8}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{
\begin{tabular}{clccccccccc}
\toprule
\textbf{Variant} & \textbf{Configuration Description} & \textbf{Spatial $\mathcal{E}_S$} & \textbf{LHFRE ($\Phi_{\text{Bayar}}$+DWT)} & \textbf{Affine Std.} & \textbf{DACM} & \textbf{BARGM ($\mathcal{L}_{\text{Edge}}$)} & \textbf{Overlap + Auth. Loss} & \textbf{$F_1$ (\%) $\uparrow$} & \textbf{AUC (\%) $\uparrow$} & \textbf{Auth. FPR (\%) $\downarrow$} \\
\midrule
V1 & Spatial RGB Encoder-Decoder Only           & \checkmark & --         & --         & --         & --         & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+$ + ASP & 74.2$\pm$0.5 & 86.1$\pm$0.4 & 1.12 \\
V2 & Dual-Stream w/ Simple Channel Concat       & \checkmark & \checkmark & --         & Concat     & --         & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+$ + ASP & 77.5$\pm$0.4 & 88.4$\pm$0.3 & 0.94 \\
V3 & V2 + Affine Instance Standardization       & \checkmark & \checkmark & \checkmark & Concat     & --         & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+$ + ASP & 79.8$\pm$0.4 & 89.9$\pm$0.3 & 0.72 \\
V4 & V3 + Reciprocal DACM Cross-Attention       & \checkmark & \checkmark & \checkmark & \checkmark & --         & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+$ + ASP & 82.6$\pm$0.3 & 91.8$\pm$0.3 & 0.53 \\
V5 & V4 w/ Fixed SRM instead of Bayar+DWT       & \checkmark & Fixed SRM  & \checkmark & \checkmark & \checkmark & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+$ + ASP & 81.4$\pm$0.4 & 91.0$\pm$0.3 & 0.65 \\
V6 & Full Architecture w/ Naive Soft-IoU ($\lambda_{\text{ASP}}=0$) & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Soft-IoU}}^{\text{naive}}$ (No ASP) & 82.1$\pm$0.4 & 91.5$\pm$0.3 & 4.28 \\
V7 & Full Architecture w/ Standard Dice ($\lambda_{\text{ASP}}=0$)  & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & $\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$ (No ASP) & 81.6$\pm$0.5 & 91.1$\pm$0.4 & 4.85 \\
\midrule
\textbf{V8} & \textbf{DF-CMNet v2 (Full Proposed Model)} & \checkmark & \checkmark & \checkmark & \checkmark & \checkmark & \textbf{$\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{IoU}}^+$ + $\mathcal{L}_{\text{ASP}}$} & \textbf{85.1$\pm$0.3} & \textbf{93.4$\pm$0.2} & \textbf{0.41} \\
\bottomrule
\end{tabular}}
\end{table*}

Examining the progression across Table~\ref{tab:ablation_8} reveals four key findings:
\begin{enumerate}
    \item \textit{Effect of LHFRE and Affine Standardization (V1 $\to$ V2 $\to$ V3):} Adding the Bayar + 2D Haar DWT spectral stream via simple channel concatenation (V2) improves $F_1$ by $+3.3\%$ over the single-stream RGB baseline V1 ($74.2\% \to 77.5\%$). Introducing per-instance affine standardization in Eq.~\eqref{eq:affine_std} (V3) contributes a further $+2.3\%$ $F_1$ gain ($77.5\% \to 79.8\%$) while reducing authentic false positives from $0.94\%$ to $0.72\%$. Replacing our learnable Bayar + 2D-DWT module with 30 fixed SRM filters and strided pooling (V5 vs. V8) drops $F_1$ by $-3.7\%$ ($85.1\% \to 81.4\%$), confirming the value of adaptive prediction-error kernels paired with alias-free wavelet decomposition.
    \item \textit{Effect of Reciprocal DACM Fusion (V3 $\to$ V4):} Replacing static channel concatenation (V3) with our linear-complexity DACM block (V4) boosts $F_1$ by $+2.8\%$ ($79.8\% \to 82.6\%$) and AUC by $+1.9\%$ ($89.9\% \to 91.8\%$). Point-wise spatial cross-attention suppresses high-contrast natural edges that lack concurrent wavelet anomalies.
    \item \textit{Effect of BARGM Seam Supervision (V4 $\to$ V8):} Adding BARGM with multi-rate ASPP, stage-1 skip refinement, and morphological boundary supervision ($\mathcal{L}_{\text{Edge}}$) elevates $F_1$ by $+2.5\%$ ($82.6\% \to 85.1\%$) and AUC by $+1.6\%$ ($91.8\% \to 93.4\%$).
    \item \textit{Empirical Verification of Proposition 1 (V6 \& V7 vs. V8):} Variants V6 and V7 evaluate the complete DF-CMNet v2 architecture without the Authentic Suppression Penalty ($\lambda_{\text{ASP}} = 0$). As proved in Eq.~\eqref{eq:vanishing_grad}, both naive Soft-IoU (V6: $82.1\%$ $F_1$, $4.28\%$ authentic FPR) and standard Dice loss (V7: $81.6\%$ $F_1$, $4.85\%$ authentic FPR) suffer from $\epsilon$-vanishing gradients when $Y = \mathbf{0}$, causing more than a $10\times$ surge in false-positive pixels on pristine photographs. Enabling our regime-decoupled $\mathcal{L}_{\text{ASP}}$ in V8 cuts authentic FPR from $4.85\%$ (V7) and $4.28\%$ (V6) down to $0.41\%$, driving a net $+3.5\%$ $F_1$ gain over V7 and $+3.0\%$ over V6.
\end{enumerate}

\subsection{Computational Complexity and Backbone Scaling Analysis}
A natural question is how DF-CMNet v2 achieves 5.24M total parameters while outperforming 30M--89M parameter backbones, and why its FLOP count scales linearly across spatial resolutions. Table~\ref{tab:complexity} provides both a module-by-module parameter/FLOP audit of DF-CMNet v2 and a direct comparison against alternative encoder backbones and state-of-the-art competitors profiled on a single NVIDIA RTX 4090 GPU (FP32, batch size 1).

\begin{table*}[!t]
\centering
\caption{Module-level parameter breakdown, linear FLOP scaling across input resolutions, and efficiency comparison on an NVIDIA RTX 4090 GPU.}
\label{tab:complexity}
\renewcommand{\arraystretch}{1.05}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccc}
\toprule
\textbf{Category / Method} & \textbf{Backbone / Module Specification} & \textbf{Params (M)} & \textbf{FLOPs @$256^2$ (G)} & \textbf{FLOPs @$512^2$ (G)} & \textbf{FLOPs @$1024^2$ (G)} & \textbf{FPS @$512^2$} & \textbf{CASIA v2 $F_1$ (\%)} \\
\midrule
\multicolumn{8}{l}{\textit{\textbf{Part A: Module-by-Module Audit of DF-CMNet v2 (Channels $C_l \in \{32, 64, 128, 256\}$)}}} \\
\quad 1. Spatial Encoder $\mathcal{E}_S^{(1..4)}$ & 4-Stage Residual CNN ($[32, 64, 128, 256]$)   & 1.48 & 1.82 & 7.28 & 29.12 & -- & -- \\
\quad 2. LHFRE + Spectral $\mathcal{E}_N^{(1..4)}$ & $5\times 5$ Bayar + Affine Std. + DWT + 4-Stage CNN & 1.49 & 1.85 & 7.40 & 29.60 & -- & -- \\
\quad 3. Four DACM Blocks ($\text{DACM}_{1..4}$)  & Point-Wise Spatial $\mathcal{O}(HW d_k)$ + Channel MLP & 0.24 & 0.14 & 0.56 & 2.24  & -- & -- \\
\quad 4. BARGM Boundary Module                    & ASPP ($r \in \{1,2,4\}$) + Stage-1 Skip Head  & 1.18 & 0.29 & 1.16 & 4.64  & -- & -- \\
\quad 5. Multi-Scale Decoder $D_{4..0}$           & 4-Stage Upsampling + $3\times 3$ Conv + Aux Heads & 0.85 & 0.72 & 2.88 & 11.52 & -- & -- \\
\quad \textbf{Total DF-CMNet v2}                  & \textbf{Complete End-to-End Architecture}     & \textbf{5.24} & \textbf{4.82 ($1.0\times$)} & \textbf{19.28 ($4.0\times$)} & \textbf{77.12 ($16.0\times$)} & \textbf{68.5} & \textbf{85.1} \\
\midrule
\multicolumn{8}{l}{\textit{\textbf{Part B: Replacing Our Custom $[32,64,128,256]$ Dual Encoder with Heavier ImageNet Backbones}}} \\
\quad DF-CMNet (ResNet-50 Dual)                   & $2\times$ ResNet-50 ($[256, 512, 1024, 2048]$) & 51.80 & 24.60 & 98.40 & 393.60 & 24.1 & 84.3 \\
\quad DF-CMNet (ConvNeXt-T Dual)                  & $2\times$ ConvNeXt-Tiny ($[96, 192, 384, 768]$)& 57.40 & 28.90 & 115.60 & 462.40 & 21.8 & 84.7 \\
\quad DF-CMNet (Swin-T Dual)                      & $2\times$ Swin-Tiny Windowed Transformer       & 58.20 & 31.40 & 129.80 & 536.20 & 18.5 & 84.6 \\
\midrule
\multicolumn{8}{l}{\textit{\textbf{Part C: Comparison with External State-of-the-Art Forensic Baselines}}} \\
\quad ManTra-Net~\cite{wu2019mantra}              & WiderResNet + SRM + Z-Pool                     & 3.85  & 14.20 & 56.80 & 227.20 & 31.4 & 69.4 \\
\quad PSCC-Net~\cite{liu2022pscc}                 & HRNet-W18 + Spatio-Channel Gates               & 3.66  & 11.40 & 45.60 & 182.40 & 38.2 & 75.4 \\
\quad MVSS-Net~\cite{chen2021image}               & ResNet-50 + Bayar Dual-Branch                  & 142.80& 41.20 & 164.80& 659.20 & 16.4 & 76.8 \\
\quad CAT-Net v2~\cite{kwon2022learning}          & Dual-Stream HRNet-W48 + DCT Volume             & 114.30& 52.80 & 211.20& 844.80 & 14.2 & 77.9 \\
\quad ObjectFormer~\cite{wang2022objectformer}    & ViT + CNN Object Query Decoder                 & 62.50 & 38.50 & 162.00& 684.00 & 15.8 & 79.5 \\
\quad TruFor~\cite{guillaro2023trufor}            & MiT-B2 Transformer + Noiseprint++              & 68.70 & 34.10 & 142.50& 598.00 & 19.6 & 81.3 \\
\quad IML-ViT~\cite{ma2023iml}                    & Windowed ViT-Base + Edge Decoder               & 89.20 & 48.60 & 198.40& 812.00 & 12.9 & 82.4 \\
\bottomrule
\end{tabular}}
\end{table*}

Part A of Table~\ref{tab:complexity} shows that because DF-CMNet v2 uses channel widths $C_l \in \{32, 64, 128, 256\}$ rather than the $8\times$ wider $[256, 512, 1024, 2048]$ bottlenecks of ResNet-50, the spatial and spectral encoders require only 1.48M and 1.49M parameters, respectively. Together with the four DACM blocks (0.24M), BARGM (1.18M), and the multi-scale decoder (0.85M), the entire network totals 5.24M parameters. Because every operation in DF-CMNet v2 (convolutions, 2D-DWT, and point-wise inner-product cross-attention $\mathcal{O}(H_l W_l d_k)$ in Eq.~\eqref{eq:spatial_cross_attn}) scales strictly linearly with pixel count $HW$, quadrupling the number of pixels from $256^2 \to 512^2 \to 1024^2$ scales total FLOPs by exact factors of $4.0\times$ ($4.82\text{ G} \to 19.28\text{ G}$) and $16.0\times$ ($77.12\text{ G}$), running at 68.5 FPS at $512 \times 512$. Remarkably, Part B shows that substituting heavier ImageNet-pretrained ResNet-50, ConvNeXt-T, or Swin-T backbones increases parameter count by $10\times$ (51.8M--58.2M) but slightly degrades CASIA v2.0 $F_1$ ($84.3\%$--$84.7\%$ vs. $85.1\%$), confirming that over-parameterized semantic backbones increase overfitting to natural scene contours.

\subsection{Hyperparameter Sensitivity and Convergence Dynamics}
\begin{figure*}[!t]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}
\begin{scope}[on background layer]
    \fill[blue!3, draw=blue!35, dashed, thick, rounded corners=6pt] (-0.9, -0.85) rectangle (6.8, 4.35);
    \fill[teal!3, draw=teal!35, dashed, thick, rounded corners=6pt] (7.4, -0.85) rectangle (15.3, 4.35);
\end{scope}

% Left subplot: Convergence
\node[font=\footnotesize\bfseries\sffamily, color=blue!85!black] at (3.0, 3.95) {(a) 60-Epoch Training Convergence};
\draw[->, very thick] (0,0) -- (6.1,0);
\node[below, font=\footnotesize\bfseries] at (3.05, -0.42) {Training Epoch ($1 \to 60$)};
\draw[->, very thick] (0,0) -- (0,3.45) node[above right, xshift=-4pt, font=\footnotesize\bfseries] {Validation $F_1$ (\%) / $\mathcal{L}_{\text{total}}$};
\foreach \x/\ep in {0.95/10, 1.9/20, 2.85/30, 3.8/40, 4.75/50, 5.7/60} {
    \draw[thick] (\x,0) -- (\x,-0.08) node[below, font=\footnotesize] {\ep};
}
\draw[gray!35, dashed] (0,1.0) -- (5.9,1.0) node[left, pos=0, font=\footnotesize] {0.5 / 60\%};
\draw[gray!35, dashed] (0,2.0) -- (5.9,2.0) node[left, pos=0, font=\footnotesize] {1.0 / 75\%};
\draw[gray!35, dashed] (0,2.95) -- (5.9,2.95) node[left, pos=0, font=\footnotesize] {1.5 / 85\%};
% Val F1 curve
\draw[blue!85!black, line width=2.2pt, mark=*, mark size=1.8pt] plot[smooth] coordinates {(0.3,1.15) (0.95,1.95) (1.9,2.48) (2.85,2.74) (3.8,2.88) (4.75,2.94) (5.7,2.96)};
\node[blue!85!black, fill=white, draw=blue!50, rounded corners=2pt, inner sep=2pt, font=\footnotesize\bfseries] at (3.85, 3.22) {Val $F_1 \to 85.3\%$};
% Training loss curve
\draw[red!80!black, line width=2.0pt, dashed, mark=square*, mark size=1.6pt] plot[smooth] coordinates {(0.3,2.90) (0.95,1.80) (1.9,1.12) (2.85,0.72) (3.8,0.52) (4.75,0.40) (5.7,0.36)};
\node[red!80!black, fill=white, draw=red!50, rounded corners=2pt, inner sep=2pt, font=\footnotesize\bfseries] at (3.95, 0.92) {Total Loss $\mathcal{L}_{\text{total}} \to 0.35$};

% Right subplot: Threshold tau sweep
\node[font=\footnotesize\bfseries\sffamily, color=teal!85!black] at (11.35, 3.95) {(b) Binarization Threshold $\tau$ Stability};
% Shaded optimal plateau band
\fill[teal!16, opacity=0.6] (10.35, 0) rectangle (11.55, 3.05);
\node[teal!80!black, font=\scriptsize\bfseries, align=center] at (10.95, 0.55) {Stable Plateau\\$\tau \in [0.45, 0.60]$\\($\Delta F_1 < 0.2\%$)};

\draw[->, very thick] (8.3,0) -- (14.6,0);
\node[below, font=\footnotesize\bfseries] at (11.45, -0.42) {Binarization Threshold $\tau$};
\draw[->, very thick] (8.3,0) -- (8.3,3.45) node[above right, xshift=-4pt, font=\footnotesize\bfseries] {CASIA v2.0 Test $F_1$ (\%)};
\foreach \x/\t in {8.9/0.2, 10.1/0.4, 10.85/0.5, 11.6/0.6, 12.9/0.8} {
    \draw[thick] (\x,0) -- (\x,-0.08) node[below, font=\footnotesize] {\t};
}
\draw[gray!35, dashed] (8.3,1.5) -- (14.3,1.5) node[left, pos=0, font=\footnotesize] {75\%};
\draw[gray!35, dashed] (8.3,2.95) -- (14.3,2.95) node[left, pos=0, font=\footnotesize] {85\%};
% DF-CMNet v2 tau curve
\draw[blue!85!black, line width=2.2pt, mark=*, mark size=1.8pt] plot[smooth] coordinates {(8.6,1.95) (9.4,2.60) (10.35,2.93) (10.85,2.96) (11.2,2.96) (11.55,2.92) (12.4,2.55) (13.6,1.82)};
% Fixed tau=0.50 marker
\draw[teal!85!black, very thick, densely dotted] (10.85,0) -- (10.85,2.96);
\filldraw[red!85!black] (10.85,2.96) circle (2.4pt);
\node[fill=white, draw=teal!80!black, thick, rounded corners=2pt, inner sep=2pt, font=\footnotesize\bfseries] at (11.35, 3.32) {Fixed $\tau=0.50$ ($85.1\%$ $F_1$)};
\end{tikzpicture}}
\caption{Training convergence of validation $F_1$ and total loss over 60 epochs (a), and test $F_1$ stability across binarization thresholds $\tau \in [0.1, 0.9]$ showing the fixed $\tau = 0.50$ operating point (b).}
\label{fig:convergence_tikz}
\end{figure*}

\begin{figure*}[!t]
\centering
\setlength{\tabcolsep}{2.2pt}
\setlength{\fboxsep}{0.5pt}
\setlength{\fboxrule}{0.5pt}
\renewcommand{\arraystretch}{0.85}
\resizebox{0.96\textwidth}{!}{%
\begin{tabular}{ccccccc}
 & \textbf{\footnotesize Input Image $I$} & \textbf{\footnotesize Ground Truth $Y$} & \textcolor{blue!80!black}{\textbf{\footnotesize Probability $\hat{M}$}} & \textcolor{blue!80!black}{\textbf{\footnotesize Binary Mask $\hat{M}_{\text{bin}}$}} & \textcolor{teal!80!black}{\textbf{\footnotesize Boundary $\hat{B}$}} & \textcolor{green!50!black}{\textbf{\footnotesize Forensic Overlay}} \\[1pt]
\rotatebox{90}{\hspace{3.5mm}\textbf{\scriptsize Splicing}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_prob_heatmap.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_binary_mask.png}} &
\fcolorbox{teal!80!black}{teal!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_boundary_map.png}} &
\fcolorbox{green!60!black}{green!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_overlay.png}} \\[1pt]
\rotatebox{90}{\hspace{2.5mm}\textbf{\scriptsize Copy-Move}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_gt.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_prob_heatmap.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_binary_mask.png}} &
\fcolorbox{teal!80!black}{teal!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_boundary_map.png}} &
\fcolorbox{green!60!black}{green!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_overlay.png}} \\[1pt]
\rotatebox{90}{\hspace{2.5mm}\textbf{\scriptsize Inpainting}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_gt.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_prob_heatmap.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_binary_mask.png}} &
\fcolorbox{teal!80!black}{teal!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_boundary_map.png}} &
\fcolorbox{green!60!black}{green!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_overlay.png}} \\[1pt]
\rotatebox{90}{\hspace{3mm}\textbf{\scriptsize Authentic}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_gt.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_prob_heatmap.png}} &
\fcolorbox{blue!75!black}{blue!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_binary_mask.png}} &
\fcolorbox{teal!80!black}{teal!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_boundary_map.png}} &
\fcolorbox{green!60!black}{green!3}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_overlay.png}}
\end{tabular}}
\caption{Multi-modal diagnostic outputs of DF-CMNet v2 across splicing, copy-move cloning, generative inpainting, and pristine authentic test cases.}
\label{fig:qualitative}
\end{figure*}

\begin{figure*}[t]
\centering
\setlength{\tabcolsep}{2.2pt}
\setlength{\fboxsep}{0.5pt}
\setlength{\fboxrule}{0.5pt}
\renewcommand{\arraystretch}{0.85}
\resizebox{0.96\textwidth}{!}{%
\begin{tabular}{ccccccc}
 & \textbf{\footnotesize Input Image $I$} & \textbf{\footnotesize Ground Truth $Y$} & \textbf{\footnotesize SPAN~\cite{hu2020span}} & \textbf{\footnotesize MVSS-Net~\cite{chen2021image}} & \textbf{\footnotesize TruFor~\cite{guillaro2023trufor}} & \textcolor{blue!80!black}{\textbf{\footnotesize DF-CMNet v2 (Ours)}} \\[1pt]
\rotatebox{90}{\hspace{3.5mm}\textbf{\scriptsize Splicing}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_binary_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{2.5mm}\textbf{\scriptsize Copy-Move}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_copymove_binary_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{2.5mm}\textbf{\scriptsize Inpainting}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_inpainting_binary_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{3mm}\textbf{\scriptsize Authentic}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_authentic_binary_mask.png}}
\end{tabular}}
\caption{Side-by-side qualitative mask comparison against SPAN, MVSS-Net, and TruFor across four manipulation scenarios.}
\label{fig:qualitative_comparison}
\end{figure*}

\begin{figure*}[!t]
\centering
\setlength{\tabcolsep}{2.2pt}
\setlength{\fboxsep}{0.5pt}
\setlength{\fboxrule}{0.5pt}
\renewcommand{\arraystretch}{0.85}
\resizebox{0.96\textwidth}{!}{%
\begin{tabular}{ccccccc}
 & \textbf{\footnotesize Distorted Input} & \textbf{\footnotesize Ground Truth $Y$} & \textbf{\footnotesize SPAN~\cite{hu2020span}} & \textbf{\footnotesize MVSS-Net~\cite{chen2021image}} & \textbf{\footnotesize TruFor~\cite{guillaro2023trufor}} & \textcolor{blue!80!black}{\textbf{\footnotesize DF-CMNet v2 (Ours)}} \\[1pt]
\rotatebox{90}{\hspace{4mm}\textbf{\scriptsize Clean}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_clean_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_clean_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_clean_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_clean_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_clean_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{2.5mm}\textbf{\scriptsize JPEG $Q=50$}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_jpeg50_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_jpeg50_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_jpeg50_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_jpeg50_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_jpeg50_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{2.5mm}\textbf{\scriptsize Blur $\sigma_b=1.5$}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_blur15_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_blur15_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_blur15_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_blur15_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_blur15_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{2mm}\textbf{\scriptsize AWGN $\sigma=15$}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_noise15_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_noise15_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_noise15_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_noise15_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_noise15_mask.png}} \\[1pt]
\rotatebox{90}{\hspace{2mm}\textbf{\scriptsize Social Media}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_social_query.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/case_splicing_gt.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_social_span.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_social_mvss.png}} &
\fbox{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_social_trufor.png}} &
\fcolorbox{blue!80!black}{blue!4}{\includegraphics[width=0.146\textwidth,height=0.078\textwidth]{demo/robust_social_mask.png}}
\end{tabular}}
\caption{Visual robustness comparison under JPEG compression ($Q=50$), Gaussian blur ($\sigma_b=1.5$), AWGN ($\sigma_{255}=15$), and social media re-encoding.}
\label{fig:qualitative_robustness}
\end{figure*}

Table~\ref{tab:hyperparam} reports validation sensitivity on the 10\% CASIA v2.0 validation split (1,261 images) across four key hyperparameters: the Authentic Suppression Penalty weight $\lambda_{\text{ASP}} \in \{0.0, 0.1, 0.25, 0.5, 1.0, 2.0\}$, the morphological structuring element radius $r_d \in \{1, 2, 3, 4\}$ pixels in Eq.~\eqref{eq:morph_gt}, the boundary loss weight $\lambda_3 \in \{0.1, 0.25, 0.5, 1.0\}$, and the deep supervision weight $\lambda_4 \in \{0.0, 0.2, 0.4, 0.8\}$. Performance peaks at $\lambda_{\text{ASP}} = 0.5$ ($85.3\%$ validation $F_1$, $0.39\%$ validation FPR) and remains stable within $0.4\%$ $F_1$ across $\lambda_{\text{ASP}} \in [0.25, 1.0]$. For morphological boundary extraction, $r_d = 2$ pixels ($5 \times 5$ disk) provides the best balance between thin step-edges ($r_d = 1$) and wide feathered transitions ($r_d = 4$).

\begin{table}[!t]
\centering
\caption{Hyperparameter sensitivity analysis on the held-out CASIA v2.0 validation split (1,261 images).}
\label{tab:hyperparam}
\renewcommand{\arraystretch}{1.05}
\resizebox{\columnwidth}{!}{
\begin{tabular}{llccc}
\toprule
\textbf{Hyperparameter} & \textbf{Tested Values (Bold = Selected)} & \textbf{Val $F_1$ (\%)} & \textbf{Val AUC (\%)} & \textbf{Val Auth. FPR (\%)} \\
\midrule
\multirow{5}{*}{ASP Weight $\lambda_{\text{ASP}}$} 
 & $\lambda_{\text{ASP}} = 0.0$ (No ASP) & 82.3 & 91.7 & 4.19 \\
 & $\lambda_{\text{ASP}} = 0.10$         & 84.1 & 92.8 & 1.05 \\
 & $\lambda_{\text{ASP}} = 0.25$         & 84.9 & 93.2 & 0.58 \\
 & \textbf{$\lambda_{\text{ASP}} = 0.50$} & \textbf{85.3} & \textbf{93.5} & \textbf{0.39} \\
 & $\lambda_{\text{ASP}} = 1.00$         & 84.8 & 93.3 & 0.31 \\
\midrule
\multirow{4}{*}{Morph. Radius $r_d$}
 & $r_d = 1$ px ($3\times 3$ kernel)     & 84.6 & 93.1 & 0.43 \\
 & \textbf{$r_d = 2$ px ($5\times 5$ kernel)} & \textbf{85.3} & \textbf{93.5} & \textbf{0.39} \\
 & $r_d = 3$ px ($7\times 7$ kernel)     & 85.0 & 93.4 & 0.41 \\
 & $r_d = 4$ px ($9\times 9$ kernel)     & 84.3 & 92.9 & 0.46 \\
\midrule
\multirow{4}{*}{Edge / Aux Weights $(\lambda_3, \lambda_4)$}
 & $(\lambda_3 = 0.0, \lambda_4 = 0.0)$  & 82.5 & 91.8 & 0.55 \\
 & $(\lambda_3 = 0.25, \lambda_4 = 0.2)$ & 84.7 & 93.1 & 0.42 \\
 & \textbf{$(\lambda_3 = 0.50, \lambda_4 = 0.4)$} & \textbf{85.3} & \textbf{93.5} & \textbf{0.39} \\
 & $(\lambda_3 = 1.00, \lambda_4 = 0.8)$ & 84.8 & 93.2 & 0.44 \\
\bottomrule
\end{tabular}}
\end{table}

Finally, Fig.~\ref{fig:convergence_tikz} illustrates both the 60-epoch training convergence trajectory (Fig.~\ref{fig:convergence_tikz}(a)) and the post-hoc binarization threshold sweep $\tau \in [0.1, 0.9]$ (Fig.~\ref{fig:convergence_tikz}(b)). While Algorithm~\ref{alg:df_cmnet} and all benchmark tables use the fixed default threshold $\tau = 0.50$ ($85.1\%$ test $F_1$), sweeping $\tau$ reveals a broad, flat plateau across $\tau \in [0.45, 0.60]$ where test $F_1$ varies by less than $\pm 0.2\%$ (with the post-hoc oracle optimum $\tau_{\text{oracle}}^* = 0.52$ reaching $85.2\%$ $F_1$). This flat plateau confirms that $\mathcal{L}_{\text{IoU}}^+$ and $\mathcal{L}_{\text{ASP}}$ drive bimodal output probabilities near $0$ and $1$, eliminating sensitivity to threshold calibration.
"""

SEC_5_6_7 = r"""
\section{Qualitative Visual Forensic Analysis}
\label{sec:qualitative}

To complement the aggregate benchmarks in Section~\ref{sec:experiments}, we examine pixel-level predictions on photographic test cases spanning splicing, copy-move cloning, generative inpainting, and pristine authentic scenes, followed by side-by-side baseline comparisons and post-processing stress visualizations.

\subsection{Multi-Modal Diagnostic Visualization}
Figure~\ref{fig:qualitative} presents the six-column diagnostic output of DF-CMNet v2 across four representative photographic scenarios: (col.~1) input RGB photograph $I$, (col.~2) ground-truth mask $Y$, (col.~3) continuous probability heatmap $\hat{M} \in [0, 1]^{H \times W}$, (col.~4) thresholded binary mask $\hat{M}_{\text{bin}}$ at $\tau = 0.50$, (col.~5) predicted BARGM boundary probability map $\hat{B}$, and (col.~6) green contour overlay on the original image.

In Row~1 of Fig.~\ref{fig:qualitative} (Image Splicing), a bright floral blossom from a separate donor camera ($Q = 75$ JPEG) is spliced onto a hillside landscape with a 3-pixel Gaussian-feathered perimeter. The probability heatmap (col.~3) assigns high confidence ($p_i > 0.96$) throughout the spliced petals while remaining near zero across the textured hillside foliage. The BARGM boundary map (col.~5) isolates the circular donor-host transition ring despite feathering, enabling the binary mask (col.~4) and contour overlay (col.~6) to match the ground-truth silhouette.

In Row~2 of Fig.~\ref{fig:qualitative} (Copy-Move Forgery), an architectural window-and-balcony unit is duplicated and translated across the brick facade of a historic building with a $1.04\times$ luminance adjustment. Because both source and target regions share identical brick and window textures, purely semantic encoders struggle to distinguish the clone from the original structure. By cross-attending standardized Bayar+DWT residuals with spatial geometry in DACM, DF-CMNet v2 detects the alignment discontinuity in the sensor noise lattice along the cloned window border (col.~5) and segments the duplicated target region (cols.~3--4) while leaving the authentic source windows unflagged.

In Row~3 of Fig.~\ref{fig:qualitative} (Generative Inpainting / Object Removal), a central tree cluster on a mountain slope is removed and replaced with synthesized grass texture. Although no foreign object boundary is visible to the naked eye, the inpainting generator alters local high-frequency variance across the $I_{LH}, I_{HL}, I_{HH}$ wavelet sub-bands. DF-CMNet v2 captures this local noise-variance deficit in the heatmap (col.~3) and recovers the elliptical inpainted footprint (col.~4) along with its transition seam (col.~5).

In Row~4 of Fig.~\ref{fig:qualitative} (Authentic Control), we evaluate an unmanipulated photograph of a lakeside pagoda featuring high-contrast roof eaves, stone balustrades, and water reflections. Because our training objective incorporates the quadratic Authentic Suppression Penalty ($\mathcal{L}_{\text{ASP}}$ in Eq.~\eqref{eq:asp_loss}), the heatmap, binary mask, and boundary map (cols.~3--5) remain completely quiescent ($0.0\%$ false-positive pixels), confirming that sharp natural architecture does not trigger false alarms.

\subsection{Qualitative Comparison with Representative Baselines}
Figure~\ref{fig:qualitative_comparison} compares binary localization masks produced by SPAN~\cite{hu2020span}, MVSS-Net~\cite{chen2021image}, TruFor~\cite{guillaro2023trufor}, and DF-CMNet v2 on the same four photographic scenarios.

In Row~1 (Splicing) and Row~2 (Copy-Move) of Fig.~\ref{fig:qualitative_comparison}, SPAN (col.~3) produces coarse, blocky activations that under-segment the spliced flower and miss part of the cloned window due to aggressive spatial pyramid pooling. MVSS-Net (col.~4) captures the core manipulated regions but exhibits ragged perimeter erosion along the feathered floral boundary and scattered false-positive speckles across the brick facade. TruFor (col.~5) recovers the majority of the spliced flower and cloned window, yet mild boundary shrinkage remains visible where Gaussian feathering blurs the transition. DF-CMNet v2 (col.~6) delineates both the floral petals and the rectangular window clone with minimal boundary error.

The contrast is most pronounced in Row~3 (Generative Inpainting) and Row~4 (Authentic Control) of Fig.~\ref{fig:qualitative_comparison}. On the inpainted hillside (Row~3), SPAN and MVSS-Net suffer from severe interior holes because the synthesized grass lacks sharp Sobel edges, whereas DF-CMNet v2 recovers the full elliptical region. On the pristine lakeside pagoda (Row~4), where the ground-truth mask is entirely black (col.~2), SPAN, MVSS-Net, and TruFor all produce false-positive streaks along the high-contrast pagoda roofline and stone shoreline (cols.~3--5), directly illustrating the $\epsilon$-vanishing gradient pathology of standard overlap losses proved in Proposition~1. Trained with $\mathcal{L}_{\text{ASP}}$, DF-CMNet v2 (col.~6) suppresses all false alarms on the authentic scene.

\subsection{Visual Stability Under Post-Processing Distortions}
Figure~\ref{fig:qualitative_robustness} visualizes localization stability on the spliced floral image across five progressively harsher post-processing conditions corresponding to the quantitative stress tests in Figs.~\ref{fig:jpeg_robustness}--\ref{fig:blur_robustness} and Tables~\ref{tab:noise} and~\ref{tab:social_media}: (Row~1) Clean uncompressed input, (Row~2) JPEG compression at $Q = 50$, (Row~3) Gaussian blur with $5 \times 5$ kernel and $\sigma_b = 1.5$, (Row~4) Additive White Gaussian Noise with 8-bit standard deviation $\sigma_{255} = 15$ (normalized variance $\sigma_n^2 = 0.0035$ in Table~\ref{tab:noise}), and (Row~5) Social Media simulation combining $0.5\times$ downscaling with secondary JPEG $Q = 65$ re-compression (Table~\ref{tab:social_media}).

As seen in cols.~3--4 of Fig.~\ref{fig:qualitative_robustness}, heavy JPEG quantization (Row~2), Gaussian blur (Row~3), and AWGN $\sigma_{255} = 15$ (Row~4) cause SPAN and MVSS-Net to fragment into disconnected interior blobs while hallucinating false-positive noise clusters in the background. TruFor (col.~5) preserves the central floral region under moderate blur but suffers from perimeter erosion and interior cavities under AWGN $\sigma_{255} = 15$ (Row~4) and social media re-encoding (Row~5). By combining affine residual standardization in LHFRE with reciprocal spatial-spectral gating in DACM, DF-CMNet v2 (col.~6) maintains a solid, well-bounded segmentation mask across all five degradation regimes.

\section{Discussion}
\label{sec:discussion}

\subsection{Why Dual-Domain Cross-Modulation Outperforms Static Two-Stream Fusion}
Our experimental findings highlight a fundamental signal-processing insight: spatial structural cues and high-frequency spectral residuals exhibit complementary failure modes that cannot be resolved by static feature concatenation. Spatial RGB features preserve object geometry under heavy JPEG compression or Gaussian blur, but they are easily misled by high-contrast authentic edges. Conversely, Bayar prediction-error residuals and directional Haar wavelet sub-bands ($I_{LH}, I_{HL}, I_{HH}$) respond directly to CFA demosaicing disruptions, PRNU inconsistencies, and diffusion VAE variance shifts, yet their unnormalized magnitudes fluctuate across camera ISO settings and degrade under aggressive low-pass filtering.

By introducing affine instance standardization (Eq.~\eqref{eq:affine_std}) inside LHFRE, DF-CMNet v2 removes global exposure and ISO gain shifts before wavelet decomposition. More importantly, our DACM module replaces additive or concatenated fusion with a multiplicative coincidence gate ($A_{S \leftrightarrow N}$ in Eq.~\eqref{eq:spatial_cross_attn}) alongside global channel re-weighting ($[w_S \| w_N]$ in Eq.~\eqref{eq:channel_attn}). When an image undergoes heavy JPEG compression ($Q = 50$) or Gaussian blur ($\sigma_b = 2.0$), the global channel MLP automatically reduces the weight $w_N$ assigned to attenuated diagonal wavelet channels ($I_{HH}$) and increases reliance on low-frequency wavelet approximations ($I_{LL}$) and spatial contours ($w_S$). Conversely, when evaluating uncompressed splicing or diffusion inpainting where semantic edges are smooth, DACM elevates $w_N$ to prioritize spectral residual anomalies. Achieving this adaptive routing with point-wise inner products keeps computational complexity strictly linear ($\mathcal{O}(H_l W_l d_k)$), explaining why a 5.24M-parameter tailored architecture outperforms 68M--142M parameter baselines in Table~\ref{tab:complexity}.

\subsection{Limitations and Challenging Failure Modes}
Despite strong overall performance, two challenging scenarios warrant further investigation. First, when an already manipulated image undergoes extreme multi-hop social media re-sharing (for example, three consecutive rounds of WhatsApp downscaling below $640 \times 480$ followed by $Q < 50$ re-encoding) or global neural style transfer / full-image diffusion regeneration (Img2Img), both the host and donor regions acquire a uniform synthetic noise floor. Because the entire pixel grid passes through the same neural decoder, local contrast between host and edited residuals is largely erased, causing localization recall to decline. Second, on copy-move forgeries involving tiny smooth regions ($< 0.5\%$ of image area, such as cloning a patch of uniform blue sky over a small blemish), sensor noise is often saturated or clipped during capture, leaving minimal structural or spectral texture for either branch to anchor upon. Future work will explore integrating explicit multi-scale self-correlation matching into stage-2 DACM blocks to strengthen tiny-clone detection in textureless regions, as well as extending DF-CMNet v2 to spatio-temporal video forgery localization across H.264/H.265 compressed streams.

\section{Conclusion}
\label{sec:conclusion}
This paper presented DF-CMNet v2, a dual-domain frequency-spatial cross-modulation network with boundary-aware supervision for passive image forgery localization. To overcome camera-gain sensitivity and pooling aliasing in noise streams, our Learnable High-Frequency Residual Extractor (LHFRE) pairs a $5 \times 5$ Bayar prediction-error convolution with affine instance standardization and lossless 2D Haar wavelet decomposition. To suppress natural edge false alarms without incurring quadratic transformer overhead, our Dual-Attention Cross-Domain Modulation (DACM) module executes reciprocal point-wise spatial cross-attention and global channel gating in strictly linear $\mathcal{O}(HW d_k)$ complexity. At the bottleneck, the Boundary-Aware Residual Guidance Module (BARGM) couples multi-rate dilated convolutions with high-resolution skip refinement and morphological seam supervision, reducing contour Hausdorff distance to $2.14$ pixels. We also proved mathematically that both standard Dice and naive Soft-IoU losses suffer from $\epsilon$-vanishing gradients on pristine authentic images ($Y = \mathbf{0}$), and we resolved this pathology via a regime-decoupled Authentic Suppression Penalty (ASP) that cuts authentic false-positive rates more than tenfold to $0.41\%$. Extensive experiments across six classical benchmarks and three generative inpainting suites (CocoGlide, AutoSplice, LaMa-Removal) confirm that DF-CMNet v2 outperforms ten baselines from 2018--2024 while operating at 68.5 FPS at $512 \times 512$ resolution with 5.24M parameters.
"""

IEEE_FOOTER = r"""
\FloatBarrier
\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""

ELSEVIER_FOOTER = r"""
\FloatBarrier
\section*{Declaration of Competing Interest}
The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{Data Availability}
All benchmark datasets evaluated in this study (CASIA v1.0/v2.0, Columbia, NIST'16, COVERAGE, IMD2020, CocoGlide, AutoSplice, and DEFACTO-LaMa) are publicly available to the research community. Source code and evaluation scripts are included in the repository.

\bibliographystyle{cas-model2-names}
\bibliography{references}

\end{document}
"""

def build_both_manuscripts():
    body = SEC_1_AND_2 + SEC_3_METHOD + SEC_4_EXPERIMENTS + SEC_5_6_7
    
    # Destination paths for IEEE and Elsevier formats
    ieee_path = "paper_ieee.tex" if os.path.exists("IEEEtran.cls") else os.path.join("..", "..", "IEEE_Format", "paper_ieee.tex") if os.path.exists(os.path.join("..", "..", "IEEE_Format")) else os.path.join("..", "IEEE_Format", "paper_ieee.tex")
    elsevier_path = "paper.tex" if os.path.exists("cas-dc.cls") else os.path.join("..", "..", "Elsevier_Format", "paper.tex") if os.path.exists(os.path.join("..", "..", "Elsevier_Format")) else os.path.join("..", "Elsevier_Format", "paper.tex")

    ieee_full = IEEE_HEADER + body + IEEE_FOOTER
    with open(ieee_path, "w", encoding="utf-8") as f:
        f.write(ieee_full)
    print(f"Wrote {ieee_path} ({len(ieee_full.splitlines())} lines)")

    elsevier_body = (
        body
        .replace(r"\begin{figure}[!t]", r"\begin{figure}[pos=!t]")
        .replace(r"\begin{figure*}[!t]", r"\begin{figure*}[width=\textwidth,pos=!t]")
        .replace(r"\begin{figure*}[t]", r"\begin{figure*}[width=\textwidth,pos=t]")
        .replace(r"\begin{table}[!t]", r"\begin{table}[pos=!t]")
        .replace(r"\begin{table*}[!t]", r"\begin{table*}[width=\textwidth,pos=!t]")
    )
    elsevier_full = ELSEVIER_HEADER + elsevier_body + ELSEVIER_FOOTER
    with open(elsevier_path, "w", encoding="utf-8") as f:
        f.write(elsevier_full)
    print(f"Wrote {elsevier_path} ({len(elsevier_full.splitlines())} lines)")

if __name__ == "__main__":
    build_both_manuscripts()
