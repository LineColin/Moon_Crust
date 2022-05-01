# Moon_Crust
Numerical modeling of the Moon's crust formation
\documentclass[legalpaper,12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[french]{babel}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{wasysym}
\usepackage{xcolor}
\usepackage{pgf}
\usepackage{caption}
\usepackage{geometry}
\geometry{legalpaper}
\usepackage{multirow}
\usepackage{verbatim}
\usepackage{subcaption}

\usepackage{fancyhdr}
\usepackage{lastpage}
\usepackage{listings}

\title{}
\author{Colin Line}
\date{}

\fancyhead[L]{}
\fancyhead[R]{}
\fancyfoot[L]{}
\rfoot{\thepage\ }
\cfoot{  }
\pagestyle{fancy}

\renewcommand\familydefault{\sfdefault}
\renewcommand{\it}{\item[$\bullet$]}

\begin{document}
	
	\begin{center}
		\textsc{\LARGE \textbf{{Formation of the Moon's crust}}}\\[1cm]
		\textsc{\Large \textbf{Documentation for the numerical modelisation}}\\[1cm]
	\end{center}
	\noindent\hrulefill
	\newline
	\newline
	
	\section*{Description}
	\noindent\hrulefill
	
	Ce document décrit la manière dont a été réalisée la modélisation numérique du problème de formation de la croûte lunaire. Il s'agit d'un code réalisé entièrement avec \verb*|Python|. Les parties qui suivent présentent les schémas numériques utilisés, les ibliothèques nécessaires pour sont utilisation ainsi que l'utilisation de ce code.
	
	\section*{Numerical modelling}
    \noindent\hrulefill
    
	Le but de ce modèle est de résoudre l'équation d'avdevction-diffusion (ref eq.) en 1D sphérique. Pour cela, le code se base sur l'utilisation des volumes fini et d'une résolution full implicit. Les deux parties de l'équations sont résolues de manières différentes. 
	
	Diffusion: \newline
	
	Utilisation d'un volume de conrôle : 
	
	En intergant sur un volume de contrôle  $CV$: 
	\begin{equation}
		\int_{CV} \dfrac{\partial T}{\partial t}  dv = \kappa \int_{CV} \nabla (\nabla T) dv \;+ \;\int_{CV} H dv
	\end{equation}
	Theorème de la divergence :
	
	\begin{equation}
		\int_{CV} \dfrac{\partial T}{\partial t}  dv = \kappa \int_{surface}  (\nabla T) \mathbf{n} dS \;+ \;\int_{CV} H dv
	\end{equation}
	Or $\nabla T = -q$ :
	
	\begin{equation}
		\int_{CV} \dfrac{\partial T}{\partial t}  dv = \kappa \int_{surface}  (q_{e} - q_{s} )\mathbf{n} dS \;+ \;\int_{CV} H dv
	\end{equation}
	\\[1cm]
	
	
	Image
	
	$\bullet$ discretisation implicite : 
	
	\begin{equation}
		\dfrac{\partial T}{\partial t}  dv = \dfrac{T^{n+1}_{i} - T^{n}_{i}}{\Delta t} \times 4\pi r_{i}^{2} \Delta r 
	\end{equation}
	
	\begin{equation}
		q_{e} = - \kappa \dfrac{T^{n+1}_{i} - T^{n+1}_{i-1}}{\Delta r} \times 4\pi r_{i-1/2}^{2} 
	\end{equation}
	
	\begin{equation}
		q_{s} = - \kappa \dfrac{T^{n+1}_{i+1} - T^{n+1}_{i}}{\Delta r} \times 4\pi r_{i+1/2}^{2} 
	\end{equation}
	
	\begin{equation}
		H dv = H  \times 4\pi r_{i}^{2} \Delta r 
	\end{equation}
	
	Donc : 
	\begin{equation}
		\dfrac{T^{n+1}_{i} - T^{n}_{i}}{\Delta t} \times r_{i}^{2} \Delta r  = - \kappa \dfrac{T^{n+1}_{i} - T^{n+1}_{i-1}}{\Delta r} \times  r_{i-1/2}^{2} + \kappa \dfrac{T^{n+1}_{i+1} - T^{n+1}_{i}}{\Delta r} \times  r_{i+1/2}^{2} +H  \times  r_{i}^{2} \Delta r 
	\end{equation}
	
	En posant $s_{i} = \dfrac{\kappa \Delta t}{r_{i}^{2} \Delta r^{2}}$
	
	\begin{equation}
		T_{i+1}^{n+1}(-s_{i}r_{i+1/2}^{2}) + T_{i}^{n+1}(1 + s_{i}r_{i+1/2}^{2} + s_{i}r_{i-1/2}^{2}) + T_{i-1}^{n+1}(-s_{i}r_{i-1/2}^{2}) = T_{i}^{n} + \Delta t H
	\end{equation}
	
	Sous forme matricielle : \newline
	
	\begin{math}
		\begin{pmatrix}
			CL_{01} & CL_{02} &  & &  \\
			A_{i} & B_{i}& C_{i} &  & 0&\\
			&\ddots & \ddots & \ddots & &\\
			0&&A_{i}&B_{i}&C_{i}\\
			&&&CL_{I1}&CL_{I2}\\
		\end{pmatrix} 
		\begin{pmatrix}
			T_{0}^{n+1}\\
			\vdots\\
			T_{i}^{n+1}\\
			\vdots\\
			T_{I}^{n+1}
		\end{pmatrix}
		=
		\begin{pmatrix}
			T_{0}^{n}\\
			\vdots\\
			T_{i}^{n}\\
			\vdots\\
			T_{I}^{n}
		\end{pmatrix}
		+
		\begin{pmatrix}
			R_{0}\\
			\vdots\\
			0\\
			\vdots\\
			R_{I}
		\end{pmatrix}
		+ \Delta t H
	\end{math}
	
	Avec : \newline
	
	\noindent
	\begin{math}
		\left\{ \begin{matrix}
			A_{i} & = &-s_{i} r_{i+1/2}^{2}\\
			B_{i} & = & 1 + s_{i} r_{i+1/2}^{2} + s_{i} r_{i1/2}^{2}\\
			C_{i} & = & -s_{i} r_{i-1/2}^{2}\\
		\end{matrix}
		\right.
	\end{math}
	\\[1cm]
	\noindent
	\begin{math}
		\left\{ \begin{matrix}
			CL_{01} & = & 1 + 2s_{0} r_{-1/2}^{2} + s_{0}r_{1/2}^{2}\\
			CL_{02} & = & -s_{0}r_{1/2}^{2}
		\end{matrix}
		\right.
	\end{math}
	\\[1cm]
	\noindent
	\begin{math}
		\left\{ \begin{matrix}
			CL_{I1} & = & s_{I}r_{I-1/2}^{2} \\
			CL_{I2} & = &  1 + 2s_{I} r_{I+1/2}^{2} + s_{I}r_{I-1/2}^{2}
		\end{matrix}
		\right.
	\end{math}
	\\[1cm]
	et\newline
	
	\noindent
	\begin{math}
		\left\{ \begin{matrix}
			R_{0} & = & 2s_{0}T_{E}r_{-1/2}^{2}\\
			R_{I} & = & 2s_{I}T_{S}r_{I+1/2}^{2}
		\end{matrix}
		\right.
	\end{math}
	\\[1cm]
	$\rightarrow$ Résoudre problème inverse $MX = b$, avec $M$ et $b$ connues et $X$ correspond à $T^{n+1}$
	
	For the advective part, we use the Total Variation Diminishing (TVD) scheme 
	
	
	\begin{equation}
		u\dfrac{\partial \tilde{T}}{\partial \tilde{r}} \sim \dfrac{F^{+} - F^{-}}{\Delta r}
	\end{equation}
	
	Avec : 
	\\[1cm]
	\begin{math}
		F^{+} = \dfrac{u}{2}[T_{i+2}^{n}(\dfrac{-\gamma(\theta_{i+1})}{2}) + T_{i+1}^{n}(1 + \dfrac{\gamma(\theta_{i+1})}{2} +  \dfrac{\gamma(\theta_{i})}{2}) + T_{i}^{n}(1 - \dfrac{\gamma(\theta_{i})}{2})] - \dfrac{\mid u\mid}{2}[T_{i+2}^{n}(\dfrac{-\gamma(\theta_{i+1})}{2}) - T_{i+1}^{n}(1 + \dfrac{\gamma(\theta_{i+1})}{2} +  \dfrac{-\gamma(\theta_{i})}{2}) + T_{i}^{n}(\dfrac{\gamma(\theta_{i})}{2} - 1)] 
	\end{math}
	\\[1cm]
	\begin{math}
		F^{-} = \dfrac{u}{2}[T_{i+1}^{n}(\dfrac{-\gamma(\theta_{i})}{2}) + T_{i}^{n}(1 + \dfrac{\gamma(\theta_{i})}{2} +  \dfrac{\gamma(\theta_{i-1})}{2}) + T_{i-1}^{n}(1 - \dfrac{\gamma(\theta_{i-1})}{2})] - \dfrac{\mid u\mid}{2}[T_{i+1}^{n}(\dfrac{-\gamma(\theta_{i})}{2}) - T_{i}^{n}(1 + \dfrac{\gamma(\theta_{i})}{2} - \dfrac{\gamma(\theta_{i-1})}{2}) + T_{i-1}^{n}(\dfrac{\gamma(\theta_{i})}{2} - 1)] 
	\end{math}
	
	Avec : 
	\\[1cm]
	
	\begin{math}
		\theta_{i} = \dfrac{T_{i}^{n} - T_{i-1}^{n}}{T_{i+1}^{n} - T_{i}^{n}} 
	\end{math}
	\\[1cm]
	et $\gamma(\theta_{i})$ correspond au limiteur de flux au choix => Lax-Xendroff $\gamma(\theta_{i}) = \gamma(\theta_{i+1}) = \gamma(\theta_{i-1}) = 1$
	
	
	
	
	
	\section*{Prerequisites}
	\noindent\hrulefill
	
	 List of all \verb*|Python| librairies requiried
	 
	 \begin{itemize}
	 	\it \verb*|numpy|
	 	\it \verb*|mathplotlib|
	 	\it \verb*|scipy.linalg|
	 \end{itemize}
	
	\section*{Utilisation}
	
	The code is 
	
\end{document}
© 2022 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

