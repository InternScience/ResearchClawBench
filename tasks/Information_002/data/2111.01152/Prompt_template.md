<!-- # README -->
<!-- The template loosely adheres to the syntax rules of Python f-string formatting.     -->
<!-- - The {} placeholders are intended for string substitutions. -->
<!-- - The {A|B} placeholders are intended for string substitutions with either A or B. -->
<!-- - The {text|None} brackets denote optional strings text. -->
# Preamble
## Preamble
**Prompt:**  
You are a physicist helping me to construct Hamiltonian and perform Hartree-Fock step by step based on my instructions. 
You should follow the instruction strictly.
Your reply should be succinct while complete. You should not expand any unwanted content.
You will be learning background knowledge by examples if necessary.
Confirm and repeat your duty if you understand it.

## Problem-solver
**Prompt:**  
You are a physicist assisting me in constructing a Hamiltonian and/or performing Hartree-Fock calculations, adhering meticulously to my instructions. Your responses should be succinct yet comprehensive, without incorporating any unnecessary information. When needed, you will learn and understand relevant background knowledge through examples.

First, you will be presented with a background summary pertaining to the question. Following this, you will be posed with a question. Your responses should integrate information from both the background and the question.

If you encounter any inconsistencies between the information in the question and the background, prioritize the information provided in the question.

## Conversation summarizer
**Prompt:**  
Below is a conversation about a physics problem showing after ```. 
The format is 1. Background of the question. 2: Question. 3: Answer.
You should summarize the history. 
You should remove redundant information, and keep the main physics,especially, paying attention to the latex equations.
Return the summarized text. 
```
**Background**
{background}

**Question**
{question}

**Answer**
{answer}
```

# Hamiltonian construction
## Construct Kinetic Hamiltonian (continuum version, single-particle)
**Prompt:**  
You will be instructed to describe the kinetic term of Hamiltonian in {system} in the {real|momentum} space in the {single-particle|second-quantized} form.   
The degrees of freedom of the system are: {degrees_of_freedom}.  
Express the Kinetic Hamiltonian {kinetic_symbol} using {variable} which are only on the diagonal terms, and arrange the basis in the order of {order}. [Note that the sublattice degrees of freedom is suppressed for now and will be stated later]

Use the following conventions for the symbols:  
{definition_of_variables}

## Construct Kinetic Hamiltonian (continuum version, second-quantized)
**Prompt:**  
You will be instructed to describe the kinetic term of Hamiltonian in {system} in the {real|momentum} space in the {single-particle|second-quantized} form.   
The degrees of freedom of the system are: {degrees_of_freedom}.  
Express the Kinetic Hamiltonian {kinetic_symbol} using {dispersion_symbol}, {annihilation_op}, and {creation_op}, where the summation of ${k|r}$ should be running over the {entire_space|first_Brillouin_zone}.

Use the following conventions for the symbols:  
{definition_of_variables}

## Construct Kinetic Hamiltonian (lattice version)
**Prompt:**  
You will be instructed to describe the kinetic term of Hamiltonian in {system} in the {real|momentum} space in the {single-particle|second-quantized} form.   
The degrees of freedom of the system are: {degrees_of_freedom}     
The kinetic term is a tight-binding model composed of the following hopping process: 
{site i and site j with the amplitude hopping}
[You should ensure the hermiticity of the Hamiltonian]
The summation should be taken over all {degrees_of_freedom} and all {real|momentum} space positions.  
Return the Kinetic Hamiltonian {kinetic_symbol}.

Use the following conventions for the symbols:  
{definition_of_variables}

## Define each term in Kinetic Hamiltonian (continuum version)
**Prompt:**  
You will be instructed to construct each term, namely {Energy_dispersion}.  
For all energy dispersions, {Energy_dispersion}, it characterizes the {parabolic|Dirac|cos} dispersion for {electrons|holes}.   
[In addition, a shift of {momentum_shift} in the momentum ${k_symbol}$ for {shifted_Ek}, respectively.]  
You should follow the EXAMPLE below to obtain the correct energy dispersion, select the correct EXAMPLE by noticing the type of dispersion.  
Finally, in the real space, the momentum ${k_symbol}=-i \partial_{{{r_symbol}}}$. You should keep the form of ${k_symbol}$ in the Hamiltonian for short notations but should remember ${k_symbol}$ is an operator.  
You should recall that {expression_kinetic}.    
Return the expression for {Energy_dispersion} in the Kinetic Hamiltonian, and substitute it into the Kinetic Hamiltonian {kinetic_symbol}.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===  
EXAMPLE 1:  
A parabolic dispersion for electron is $E_{{\alpha}}=\frac{{\hbar^2 k^2}}{{2m_{{\alpha}}}}$, where $\alpha$ indicates the type of electron.  If there is a further shift of $q$ in the momentum $k$, the dispersion will become $E_{{\alpha}}=\frac{{\hbar^2 (k-q)^2}}{{2m_{{\alpha}}}}$.

EXAMPLE 2:
A cos dispersion is $E_{{\alpha}}=-\cos(k a / 2)$, where $\alpha$ indicates the type of particle.  If there is a further shift of $q$ in the momentum $k$, the dispersion will become $E_{{\alpha}}=-\cos((k-q) a / 2))$. However, there could be more prefactors before cos depending on the geometry of the lattice.

EXAMPLE 3:  
A dirac dispersion for electron/hole is a 2 by 2 matrix, i.e., $h_{{\theta}}(k)=-\hbar v_D |k| \begin{{pmatrix}}  0 & e^{{i(\theta_{{k}}-\theta)}}\\ e^{{-i(\theta_{{\bar{{k}}}}-\theta)}} & 0 \end{{pmatrix}}$, where $v_D$ is the Fermi velocity, $\theta$ is the twist angle, and $\theta_k$ indicates the azumith angle of $k$. If there is a further shift of $K_{{\theta}}$ in the momentum $k$, the dispersion will become $h_{{\theta}}(k)=-\hbar v_D |k-K_{{\theta}}| \begin{{pmatrix}}  0 & e^{{i(\theta_{{k-K_{{\theta}}}}-\theta)}}\\ e^{{-i(\theta_{{k-K_{{\theta}}}}-\theta)}} & 0 \end{{pmatrix}}$.


## Construct Potential Hamiltonian (continuum version)
**Prompt:**  
You will be instructed to describe the potential term of Hamiltonian {potential_symbol} in the {real|momentum} space in the {single-particle|second-quantized} form.  
The potential Hamiltonian has the same degrees of freedom as the kinetic Hamiltonian.  
The diagonal terms are {diagonal_potential}.  
The off-diagonal terms are the coupling between {potential_degrees_of_freedom}, {offdiagonal_potential}, which should be kept hermitian.  
All other terms are zero.
Express the potential Hamiltonian {potential_symbol} using {diagonal_potential} and {offdiagonal_potential}.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

## Define each term in Potential Hamiltonian (continuum version)
**Prompt:**  
You will be instructed to construct each term {potential_symbol}, namely, {Potential_variables}.  
The expression for diagonal terms are: {expression_diag}.  
The expression for off-diagonal terms are: {expression_offdiag}.  
You should recall that {expression_Potential}.  
Return the expressions for {Potential_variables}, and substitute it into the potential Hamiltonian {potential_symbol}.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

## Construct interaction Hamiltonian (real space, lattice version)
**Prompt:**  
You will be instructed to construct the interaction part of the Hamiltonian, {second_int_symbol} in the real space in the second-quantized form.   
The interacting Hamiltonian has the same degrees of freedom as the kinetic Hamiltonian {kinetic_symbol}.  
The interaction is a density-density interaction composed of the following process:
{site i and site j with the interaction strength}
The summation should be taken over all {degrees_of_freedom} and all real space positions.  
Return the interaction term {second_int_symbol} in terms of {density_symbol}.

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know): 
{definition_of_variables}

## Construct interaction Hamiltonian (momentum space)
**Prompt:**  
You will be instructed to construct the interaction part of the Hamiltonian {second_int_symbol} in the momentum space.  
The interaction Hamiltonian is a product of four parts.
The first part is the product of four operators with two creation and two annihilation operators following the normal order, namely, creation operators are before annihilation operators. You should follow the order of $1,2,2,1$ for the {index_of_operator}, and $1,2,3,4$ for the {momentum}. 
The second part is the constraint of total momentum conservation, namely the total momentum of all creation operators should be the same as that of all annihilation operators. [For each operator, the total momentum is the sum of moire reciprocal lattice $b_i$ and momentum with in the first BZ $k_i$]  
The third part is the interaction form. You should use {interaction} with $V(q)={int_form}$, where $q$ is the transferred total momentum between a creation operator and an annihilation operator with the same {index_of_operator}, namely $q=k_1-k_4$.  
The fourth part is the normalization factor, you should use {normalization_factor} here.
Finally, the summation should be running over all {index_of_operator}, and {momentum}
Return the interaction term {second_int_symbol} in terms of {op} and $V(q)$ (with $q$ expressed in terms of {momentum}).  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

## Convert from single-particle to second-quantized form, return in matrix
**Prompt:**  
You will be instructed to construct the second quantized form of the total noninteracting Hamiltonian in the {real|momentum} space.  
The noninteracting Hamiltonian in the {real|momentum} space {nonint_symbol} is the sum of Kinetic Hamiltonian {kinetic_symbol} and Potential Hamiltonian {potential_symbol}.  
To construct the second quantized form of a Hamiltonian. You should construct the creation and annihilation operators from the basis explicitly. You should follow the EXAMPLE below to convert a Hamiltonian from the single-particle form to second-quantized form.  
Finally by "total", it means you need to take a summation over the {real|momentum} space position {$r$|$k$}.   
Return the second quantized form of the total noninteracting Hamiltonian {second_nonint_symbol}  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===  
EXAMPLE:  
For a Hamiltonian $H$, where $H=\begin{{pmatrix}} H_{{a,a}} & H_{{a,b}} \\ H_{{b,a}} & H_{{b,b}} \end{{pmatrix}}$ and the order of basis is (a), (b), we can construct the creation operators $\psi_a^\dagger$  and $\psi_b^\dagger$, and the annihilation operator $\psi_a$  and $\psi_b$.  
The corresponding second quantized form is $\hat{{H}}=\vec{{\psi}}^\dagger H \vec{{\psi}}$, where $\vec{{\psi}}=\begin{{pmatrix}} \psi_a \\ \psi_b \end{{pmatrix}}$ and $\vec{{\psi}}^\dagger=\begin{{pmatrix}} \psi_a^\dagger & \psi_b^\dagger \end{{pmatrix}}$.  

## Convert from single-particle to second-quantized form, return in summation (expand the matrix)
**Prompt:**  
You will be instructed to expand the second-quantized form Hamiltonian {second_nonint_symbol} using {matrix_element_symbol} and {basis_symbol}. You should follow the EXAMPLE below to expand the Hamiltonian.  
You should use any previous knowledge to simplify it. For example, if any term of {matrix_element_symbol} is zero, you should remove it from the summation.
You should recall that {second_nonint_symbol} is {expression_second_nonint}.  
Return the expanded form of {second_nonint_symbol} after simplification.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===  
EXAMPLE:  
For a $\hat{{H}}=\vec{{\psi}}^\dagger H \vec{{\psi}}$, where $\vec{{\psi}}=\begin{{pmatrix}} \psi_a \\ \psi_b \end{{pmatrix}}$ and $\vec{{\psi}}^\dagger=\begin{{pmatrix}} \psi_a^\dagger & \psi_b^\dagger \end{{pmatrix}}$, we can expand it as  $\hat{{H}}=\sum_{{i,j=\{{a,b\}}}} \psi_i^\dagger H_{{i,j}} \psi_j$.  

## Convert noninteracting Hamiltonian in real space to momentum space (continuum version)
**Prompt:**  
You will be instructed to convert the total noninteracting Hamiltonian in the second quantized form from the basis in real space to the basis by momentum space.  
To do that, you should apply the Fourier transform to {real_creation_op} in the real space to the {momentum_creation_op} in the momentum space, which is defined as {definition_of_Fourier_Transformation}, where {real_variable} is integrated over the {entire_real|first_Brillouin_Zone}. You should follow the EXAMPLE below to apply the Fourier transform.  
Express the total noninteracting Hamiltonian {second_nonint_symbol} in terms of {momentum_creation_op}. Simplify any summation index if possible.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===  
EXAMPLE:  
Write a Hamiltonian $\hat{{H}}$ in the second quantized form, $\hat{{H}}=\int dr \psi(r)^\dagger H(r) \psi(r)$, where $r$ is integrated over the entire real space.  
Define the Fourier transform $c^\dagger(k)=\frac{{1}}{{\sqrt{{V}}}} \int \psi^\dagger(r) e^{{i k \cdot r}} dr$, where $r$ is integrated over the entire real space, and $V$ is the area of the unit cell in the real space.  
This leads to the inverse Fourier transform $\psi^\dagger(r) = \frac{{1}}{{\sqrt{{V}}}} \sum_k c^\dagger(k) e^{{-i k \cdot r}}$, where $k$ is summed over the extended Brillouin zone (i.e., the entire momentum space), $\Omega$ is the area of Brillouin zone in the momentum space.  
Thus, substitute $\psi^\dagger(r)$ and $\psi(r)$ into $\hat{{H}}$, we get  
$$\hat{{H}} = \int dr \frac{{1}}{{\sqrt{{V}}}} \sum_{{k_1}} c^\dagger(k_1) e^{{-i k_1 \cdot r}} H(r) \frac{{1}}{{\sqrt{{V}}}} \sum_{{k_2}} c(k_2) e^{{i k_2 \cdot r}} =\sum_{{k_1,k_2}} c^\dagger(k_1) \frac{{1}}{{V}} \int dr e^{{-i (k_1-k_2)\cdot r}} H(r) c(k_2) = \sum_{{k_1,k_2}} c^\dagger(k_1) H(k_1,k_2) c(k_2)$$  
, where we define the Fourier transform of $H(r)$ as $H(k_1,k_2)=\frac{{1}}{{V}} \int dr e^{{-i (k_1-k_2)\cdot r}} H(r)$.

## Convert noninteracting Hamiltonian in real space to momentum space (lattice version)
**Prompt:**  
You will be instructed to convert the noninteracting Hamiltonian {nonint_symbol} in the second quantized form from the basis in real space to the basis in momentum space. 
To do that, you should apply the Fourier transform to {real_creation_op} in the real space to the {momentum_creation_op} in the momentum space, which is defined as {definition_of_Fourier_Transformation}, where {real_variable} is integrated over all sites in the entire real space. You should follow the EXAMPLE below to apply the Fourier transform. [Note that hopping have no position dependence now.]
You should recall that {nonint_symbol} is {expression_nonint}
Express the total noninteracting Hamiltonian {nonint_symbol} in terms of {momentum_creation_op}. Simplify any summation index if possible.

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):
{definition_of_variables}

===
EXAMPLE:  
Write a Kinetic Hamiltonian $\hat{{H}}$ in the second quantized form in the real space, $\hat{{H}}=\sum_{{i,j}} t(R_i-R_j) c^\dagger(R_i) c(R_j)$, where $i,j$ are summed over the entire real space.  
Define the Fourier transform $c^\dagger(k)=\frac{{1}}{{\sqrt{{N}}}} \sum_{{i}}c^\dagger(R_i) e^{{i k \cdot R_i}}$, where $i$ is integrated over the entire real space containing $N$ unit cells, $N$ is the number of unit cells.  
This leads to the inverse Fourier transform $c^\dagger(R_i) = \frac{{1}}{{\sqrt{{N}}}} \sum_k c^\dagger(k) e^{{-i k \cdot R_i}}$, where $k$ is first Brillouin zone.  
Thus, substitute $c^\dagger(R_i)$ and $c(R_j)$ into $\hat{{H}}$, we get  
$$\hat{{H}} = \sum_{{i,j}} t(R_i-R_j) \frac{{1}}{{\sqrt{{N}}}} \sum_{{k_1}} c^\dagger(k_1) e^{{-i k_1 \cdot R_i}} \frac{{1}}{{\sqrt{{N}}}} \sum_{{k_2}} c(k_2) e^{{i k_2 \cdot R_j}} =\frac{{1}}{{N}} \sum_{{i,j}}\sum_{{k_1,k_2}} c^\dagger(k_1)  c(k_2)  e^{{-i k_1\cdot R_i}} e^{{i k_2 \cdot R_j}} t(R_i-R_j) $$
Now make a replacement by defining $n= R_i-R_j$  
The Hamiltonian become  
$$\hat{{H}}=\frac{{1}}{{N}} \sum_{{i,n}} \sum_{{k_1,k_2}} c^\dagger(k_1)  c(k_2) t(n) e^{{-i (k_1-k_2)\cdot R_i}} e^{{-i k_2 \cdot n}}$$
Because $\frac{{1}}{{N}}\sum_{{i}} e^{{-i (k_1-k_2)\cdot R_i}} = \delta(k_1,k_2)$, where $\delta(k_1,k_2)$ is the Kronecker delta function.  
therefore   
$$\hat{{H}}=\sum_{{k_1,k_2}} \sum_{{n}} t(n) e^{{-i k_2 \cdot n}} c^\dagger(k_1)  c(k_2) \delta(k_1,k_2)$$
Using the property of Kronecker delta function and sum over $k_2$, we obtain  
$$\hat{{H}}=\sum_{{k_1}} \sum_{{n}} t(n) e^{{-i k_1 \cdot n}} c^\dagger(k_1)  c(k_1) $$
For simplicity, we replace $k_1$ with $k$, we obtain  
$$\hat{{H}}=\sum_{{k}} \sum_{{n}} t(n) e^{{-i k \cdot n}} c^\dagger(k)  c(k)$$
If we define energy dispersion $E(k)=\sum_{{n}} t(n) e^{{-i k \cdot n}}$, where $n$ is the summation of all hopping pairs, the Hamiltonian in the momentum space is 
$$\hat{{H}}=\sum_{{k}} E(k) c^\dagger(k)  c(k)$$

## Convert interacting Hamiltonian in real space to momentum space (lattice version)
**Prompt:**  
You will be instructed to convert the interacting Hamiltonian, {second_int_symbol}, in the {single-particle|second-quantized} form the basis in real space to the basis in momentum space.
To do that, you should apply the Fourier transform to {real_creation_op} in the real space to the {momentum_creation_op} in the momentum space, which is defined as {definition_of_Fourier_Transformation}, where {real_variable} is integrated over all sites in the entire real space, and {momentum_var} is defined within the first Brillouin zone. You should follow the EXAMPLE below to apply the Fourier transform. [Note that interaction have no position dependence now]  
You should recall that {second_int_symbol} is {expression_int}.
Express {second_int_symbol} in terms of {momentum_creation_op}. Simplify any summation index if possible.  

===  
EXAMPLE:  
Write an interacting Hamiltonian $\hat{{H}}^{{int}}$ in the second quantized form in the real space, $\hat{{H}}^{{int}}=\sum_{{s,s'}}\sum_{{i,j}} U(R_i-R_j) c_s^\dagger(R_i) c_{{s'}}^\dagger(R_j) c_{{s'}}(R_j) c_s(R_i)$, where $i,j$ are summed over the entire real space.  
Define the Fourier transform $c_s^\dagger(k)=\frac{{1}}{{\sqrt{{N}}}} \sum_{{i}}c_s^\dagger(R_i) e^{{i k \cdot R_i}}$, where $i$ is integrated over the entire real space containing $N$ unit cells, $N$ is the number of unit cells.  
This leads to the inverse Fourier transform $c_s^\dagger(R_i) = \frac{{1}}{{\sqrt{{N}}}} \sum_k c_s^\dagger(k) e^{{-i k \cdot R_i}}$, where $k$ is summed over the first Brillouin zone.  
Thus, substitute $c^\dagger(R_i)$ and $c(R_j)$ into $\hat{{H}}^{{int}}$, we get  
$$\hat{{H}}^{{int}} = \sum_{{s,s'}}\sum_{{i,j}} U(R_i-R_j) \frac{{1}}{{\sqrt{{N}}}} \sum_{{k_1}} c_s^\dagger(k_1) e^{{-i k_1 \cdot R_i}} \frac{{1}}{{\sqrt{{N}}}} \sum_{{k_2}} c_{{s'}}^\dagger(k_2) e^{{-i k_2 \cdot R_j}} \frac{{1}}{{\sqrt{{N}}}} \sum_{{k_3}} c_{{s'}}(k_3) e^{{i k_3 \cdot R_j}} \frac{{1}}{{\sqrt{{N}}}} \sum_{{k_4}} c_s(k_4) e^{{i k_4 \cdot R_i}}=\sum_{{s,s'}}\sum_{{i,j}}\frac{{1}}{{N^2}}\sum_{{k_1,k_2,k_3,k_4}}U(R_i-R_j)c_s^\dagger(k_1)c_{{s'}}^\dagger(k_2)c_{{s'}}(k_3)c_s(k_4)e^{{-i(k_1-k_4)\cdot R_i}} e^{{-i(k_2-k_3)\cdot R_j}}$$
Now make a replacement by defining $n= R_i-R_j$  
The Hamiltonian become  
$$\hat{{H}}^{{int}}=\frac{{1}}{{N^2}} \sum_{{j,n}} \sum_{{s,s'}} \sum_{{k_1,k_2,k_3,k_4}} U(n) c_s^\dagger(k_1)c_{{s'}}^\dagger(k_2)c_{{s'}}(k_3)c_s(k_4)e^{{-i(k_1-k_4)\cdot n}} e^{{-i(k_1-k_4+k_2-k_3)\cdot r_j }}$$
Because $\frac{{1}}{{N}}\sum_{{i}} e^{{-i (k_1-k_4+k_2-k_3)\cdot R_i}} = \sum\delta(k_1-k_4+k_2-k_3,G)$, where $\delta(..,..)$ is the Kronecker delta function, and $G$ is the all reciprocal lattices in the momentum space.  
Therefore,  
$$\hat{{H}}^{{int}}=\frac{{1}}{{N}}\sum_{{s,s'}}\sum_{{k_1,k_2,k_3,k_4}} \sum_{{n}} U(n) e^{{-i (k_1-k_4) \cdot n}} c_s^\dagger(k_1)c_{{s'}}^\dagger(k_2)c_{{s'}}(k_3)c_s(k_4) \sum_{{G}} \delta(k_1-k_4+k_2-k_3,G)$$
If we define interaction in the momentum space $U(k)=\sum_{{n}} U(n) e^{{-i k \cdot n}}$, where $n$ is the summation of all hopping pairs, the interacting Hamiltonian in the momentum space is  
$$\hat{{H}}^{{int}}=\frac{{1}}{{N}}\sum_{{s,s'}}\sum_{{k_1,k_2,k_3,k_4}}  U(k_1-k_4) c_s^\dagger(k_1)c_{{s'}}^\dagger(k_2)c_{{s'}}(k_3)c_s(k_4) \sum_{{G}} \delta(k_1-k_4+k_2-k_3,G)$$

## Particle-hole transformation
**Prompt:**  
You will be instructed to perform a particle-hole transformation.  
Define a hole operator, {hole_op}, which equals {particle_op}.  
You should replace {particle_creation_op} with {hole_creation_op}, and {particle_annihilation_op} with {hole_annihilation_op}. You should follow the EXAMPLE below to apply the particle-hole transformation.  
You should recall that {expression_particle_Ham}.
Return the {second_nonint_symbol} in the hole operators.

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===  
EXAMPLE:  
Give a Hamiltonian  $\hat{{H}}=\sum_{{k_1,k_2}} c^\dagger(k_1) h(k_1,k_2) c(k_2)$ , and the particle-hole transformation as $b(k)=c^\dagger(k)$. The transformed Hamiltonian is $\hat{{H}}=\sum_{{k_1,k_2}} b(k_1) h(k_1,k_2) b^\dagger(k_2)$ 

## Simplify the Hamiltonian in the particle-hole basis
**Prompt:**  
You will be instructed to simplify the {second_nonint_symbol} in the hole basis.  
You should use canonical commutator relation for fermions to reorder the hole operator to the normal order. Normal order means that creation operators always appear before the annihilation operators.  You should follow the EXAMPLE below to simplify it to the normal order.  
Express the {second_nonint_symbol} in the normal order of {hole_op} and also make {index_1} always appear before {index_2} in the index of {op} and {Ham_op}.  
You should recall that {expression_hole_Ham}
Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===
EXAMPLE:  
For $\hat{{H}}^{{0}}= \sum_{{i,j}} b_i H_{{i,j}} b_j^\dagger$, where $b_i$ satisfies fermion statistics (anticommuting to its self-adjoint operator).  
This gives $`[b_i, b_j^\dagger]_{{+}} = \delta_{{i,j}}`$, which means $b_i b_j^\dagger=\delta_{{i,j}}-b_j^\dagger b_i$.  
Substitute it into $\hat{{H}}^{{0}}$, we have $\hat{{H}}^{{0}}=\sum_{{i,j}} (\delta_{{i,j}}-b_j^\dagger b_i) H_{{i,j}}=\sum_{{i,j}} \delta_{{i,j}} H_{{i,j}} - \sum_{{i,j}}b_j^\dagger b_i H_{{i,j}}=\sum_i H_{{i,i}} - \sum_{{i,j}}b_j^\dagger b_i H_{{i,j}}$.  
The first term is simply $\sum_i H_{{i,i}}$ by summing over the index $j$ due to $\delta_{{i,j}}$.
The second term is $-\sum_{{i,j}}b_j^\dagger b_i H_{{i,j}}$.  
Relabeling the index of $i$ and $j$ by swapping them to make it consistent with the original order of index (namely, $i$ appears before $j$ in the index of $b$ and $H$), it becomes $-\sum_{{i,j}}b_i^\dagger H_{{j,i}} b_j$.
Finally, to fix the order of the index in $H$ such that $i$ appears before $j$, we notice that $`H_{{j,i}}=(H_{{i,j}})^*`$, where $`^*`$ means complex conjugate, because the Hamiltonian is Hermitian.  
Thus, we end up in $\hat{{H}}^{{0}}=\sum_{{i,j}} b_i H_{{i,j}} b_j^\dagger=\sum_i H_{{i,i}}-\sum_{{i,j}}b_i^\dagger (H_{{i,j}})^* b_j$

# Mean-field theory
## Wick's theorem
**Prompt:**  
You will be instructed to perform a Hartree-Fock approximation to expand the interaction term, {second_int_symbol}.  
You should use Wick's theorem to expand the four-fermion term in {second_int_symbol} into quadratic terms. You should strictly follow the EXAMPLE below to expand using Wick's theorem, select the correct EXAMPLE by noticing the order of four-term product with and without ${{}}^\dagger$, and be extremely cautious about the order of the index and sign before each term.  
You should only preserve the normal terms. Here, the normal terms mean the product of a creation operator and an annihilation operator.  
You should recall that {second_int_symbol} is {expression_int}.  
Return the expanded interaction term after Hartree-Fock approximation as {Hartree_Fock_symbol}.

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

===  
EXAMPLE 1:  
For a four-fermion term $a_1^\dagger a_2^\dagger a_3 a_4$, using Wick's theorem and preserving only the normal terms. this is expanded as $a_1^\dagger a_2^\dagger a_3 a_4 = \langle a_1^\dagger a_4 \rangle a_2^\dagger a_3 + \langle a_2^\dagger a_3 \rangle a_1^\dagger a_4 - \langle a_1^\dagger a_4 \rangle \langle a_2^\dagger a_3\rangle - \langle a_1^\dagger a_3 \rangle a_2^\dagger a_4 - \langle a_2^\dagger a_4 \rangle a_1^\dagger a_3 + \langle a_1^\dagger a_3\rangle \langle a_2^\dagger a_4 \rangle$  
Be cautious about the order of the index and sign before each term here.

EXAMPLE 2:  
For a four-fermion term $a_1^\dagger a_2 a_3^\dagger a_4$, using Wick's theorem and preserving only the normal terms. this is expanded as $a_1^\dagger a_2 a_3^\dagger a_4 = \langle a_1^\dagger a_2 \rangle a_3^\dagger a_4 + \langle a_3^\dagger a_4 \rangle a_1^\dagger a_2 - \langle a_1^\dagger a_2 \rangle \langle a_3^\dagger a_4\rangle - \langle a_1^\dagger a_4 \rangle a_3^\dagger a_2 - \langle a_3^\dagger a_2 \rangle a_1^\dagger a_4 + \langle a_1^\dagger a_4\rangle \langle a_3^\dagger a_2 \rangle$  
Be cautious about the order of the index and sign before each term here.

## Extract quadratic term
**Prompt:**  
You will be instructed to extract the quadratic terms in the {Hartree_Fock_term_symbol}.  
The quadratic terms mean terms that are proportional to {bilinear_op}, which excludes terms that are solely expectations or products of expectations.  
You should only preserve the quadratic terms in {Hartree_Fock_term_symbol}, denoted as {Hartree_Fock_second_quantized_symbol}.  
You should recall that {Hartree_Fock_term_symbol} is {expression_HF}.  
Return {Hartree_Fock_second_quantized_symbol}.  

Use the following conventions for the symbols (You should also obey the conventions in all my previous prompts if you encounter undefined symbols. If you find it is never defined or has conflicts in the conventions, you should stop and let me know):  
{definition_of_variables}

# Order parameters
## Hartree term only
**Prompt:**  
You will be instructed to keep only the Hartree term in {Hartree_Fock_second_quantized_symbol}.  
Here, Hartree term only means that only the expected value in the form {expected_value_Hartree} (Note that the two indices are the same) should be the preserved. All other expected value terms should be dropped.
You should recall that {Hartree_Fock_second_quantized_symbol} is {expression_HF}  
Return the simplified Hamiltonian with {Hartree_second_quantized_symbol}.  

## Fock term only
**Prompt:**  
You will be instructed to keep only the Fock term in {Hartree_Fock_second_term_symbol}.  
Here, Fock term only means that only the expected value in the form {expected_value_Fock} (Note that the two indices are the different) should be the preserved. All other expected value terms should be dropped.
You should recall that {Hartree_Fock_second_term_symbol} is {expression_HF}  
Return the simplified Hamiltonian with {Fock_second_quantized_symbol}.

# Simplify the MF quadratic term
## Expand interaction
**Prompt:**  
You will be instructed to expand interaction term $V(q)$ in the MF quadratic term {Hartree_Fock_second_quantized_symbol}.
If you find the $V(q)$ in {Hartree_Fock_second_quantized_symbol} does not contain any momentum that is not in the summation sign. The interaction term is already expanded. No action to perform on interaction term.
Otherwise, you will expand $V(q)$ by replacing $q$ with the momentum {momentum}.
You should recall that {Hartree_Fock_second_quantized_symbol} is {expression_HF_2}.
Return {Hartree_Fock_second_quantized_symbol} with expanded interaction.

## Swap the index only
**Prompt:**  
You will be instructed to simplify the quadratic term {Hartree_Fock_second_quantized_symbol} through relabeling the index.  
The logic is that the expected value ({expected_value}) in the first Hartree term ({expression_Hartree_1}) has the same form as the quadratic operators in the second Hartree term ({expression_Hartree_2}), and vice versa. The same applies to the Fock term.  
Namely, a replacement of {relabel} is applied to ONLY the second Hartree or Fock term. You should not swap any index that is not in the summation, which includes {Unsummed_Indices}.  
This means, if you relabel the index by swapping the index in the "expected value" and "quadratic operators" in the second Hartree or Fock term, you can make the second Hartree or Fock term look identical to the first Hartree or Fock term, as long as $V(q)=V(-q)$, which is naturally satisfied in Coulomb interaction. You should follow the EXAMPLE below to simplify it through relabeling the index.  
You should recall that {Hartree_Fock_second_quantized_symbol} is {expression_HF_2}
Return the simplified {Hartree_Fock_second_quantized_symbol}.

===  
EXAMPLE:  
Given a Hamiltonian $\hat{{H}}=\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_1-k_4) (\langle c_{{d,\sigma_1}}^\dagger(k_1) c_{{d,\sigma_4}}(k_4) \rangle c_{{p,\sigma_2}}^\dagger(k_2) c_{{p,\sigma_3}}(k_3) + \langle c_{{p,\sigma_2}}^\dagger(k_2) c_{{d,\sigma_3}}(k_3) \rangle c_{{d,\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) ) \delta_{{k_1+k_2,k_3+k_4}}$, where $V(q)=V(-q)$.  
In the second term, we relabel the index to swap the index in expected value and the index in quadratic operators, namely, $\sigma_1 \leftrightarrow \sigma_2$, $\sigma_3 \leftrightarrow \sigma_4$, $k_1 \leftrightarrow k_2$, $k_3 \leftrightarrow k_4$. Important: $d$ and $p$ cannot be swapped because they are not indices in the summation.  
After the replacement, the second term becomes $\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_2-k_3) \langle c_{{p,\sigma_1}}^\dagger(k_1) c_{{p,\sigma_4}}(k_4) \rangle c_{{d,\sigma_2}}^\dagger(k_2) c_{{d,\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$.  
Note that the Kronecker dirac function $\delta_{{k_4+k_3,k_2+k_1}}$ implies $k_1+k_2=k_3+k_4$, i.e., $k_2-k_3=k_4-k_1$. Thus, the second term simplifies to $\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_4-k_1) \langle c_{{p,\sigma_1}}^\dagger(k_1) c_{{p,\sigma_4}}(k_4) \rangle c_{{d,\sigma_2}}^\dagger(k_2) c_{{d,\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$.
Because $V(q)=V(-q)$, meaning $V(k_4-k_1)=V(k_1-k_4)$, the second term further simplifies to $\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_1-k_4) \langle c_{{p,\sigma_1}}^\dagger(k_1) c_{{p,\sigma_4}}(k_4) \rangle c_{{d,\sigma_2}}^\dagger(k_2) c_{{d,\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$.   
Finally, we have the simplified Hamiltonian as  $\hat{{H}}=\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_1-k_4) (\langle c_{{d,\sigma_1}}^\dagger(k_1) c_{{d,\sigma_4}}(k_4) \rangle c_{{p,\sigma_2}}^\dagger(k_2) c_{{p,\sigma_3}}(k_3) + \langle c_{{p,\sigma_1}}^\dagger(k_1) c_{{p,\sigma_4}}(k_4) \rangle c_{{d,\sigma_2}}^\dagger(k_2) c_{{d,\sigma_3}}(k_3)) \delta_{{k_4+k_3,k_2+k_1}}$.

## Swap the index to combine Hartree and Fock terms
**Prompt:**  
You will be instructed to simplify the quadratic term {Hartree_Fock_second_quantized_symbol} through relabeling the index to combine the two Hartree/Fock term into one Hartree/Fock term.  
The logic is that the expected value ({expected_value}) in the first Hartree term ({expression_Hartree_1}) has the same form as the quadratic operators in the second Hartree term ({expression_Hartree_2}), and vice versa. The same applies to the Fock term.  
This means, if you relabel the index by swapping the index in the "expected value" and "quadratic operators" in the second Hartree term, you can make the second Hartree term look identical to the first Hartree term, as long as $V(q)=V(-q)$, which is naturally satisfied in Coulomb interaction. You should follow the EXAMPLE below to simplify it through relabeling the index.  
You should perform this trick of "relabeling the index" for both two Hartree terms and two Fock terms to reduce them to one Hartree term, and one Fock term.  
You should recall that {Hartree_Fock_second_quantized_symbol} is {expression_HF_2}.  
Return the simplified {Hartree_Fock_second_quantized_symbol} which reduces from four terms (two Hartree and two Fock terms) to only two terms (one Hartree and one Fock term)

===  
EXAMPLE:
Given a Hamiltonian $\hat{{H}}=\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_1-k_4) (\langle c_{{\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) \rangle c_{{\sigma_2}}^\dagger(k_2) c_{{\sigma_3}}(k_3) + \langle c_{{\sigma_2}}^\dagger(k_2) c_{{\sigma_3}}(k_3) \rangle c_{{\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) ) \delta_{{k_1+k_2,k_3+k_4}}$, where $V(q)=V(-q)$.  
In the second term, we relabel the index to swap the index in expected value and the index in quadratic operators, namely, $\sigma_1 \leftrightarrow \sigma_2$, $\sigma_3 \leftrightarrow \sigma_4$, $k_1 \leftrightarrow k_2$, $k_3 \leftrightarrow k_4$. After the replacement, the second term becomes $\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_2-k_3) \langle c_{{\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) \rangle c_{{\sigma_2}}^\dagger(k_2) c_{{\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$.  
Note that the Kronecker dirac function $\delta_{{k_4+k_3,k_2+k_1}}$ implies $k_1+k_2=k_3+k_4$, i.e., $k_2-k_3=k_4-k_1$. Thus, the second term simplifies to $\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_4-k_1) \langle c_{{\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) \rangle c_{{\sigma_2}}^\dagger(k_2) c_{{\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$.
Because $V(q)=V(-q)$, meaning $V(k_4-k_1)=V(k_1-k_4)$, the second term further simplifies to $\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_1-k_4) \langle c_{{\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) \rangle c_{{\sigma_2}}^\dagger(k_2) c_{{\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$. Note that this form of second term after relabeling is identical to the first term.  
Finally, we have the simplified Hamiltonian as  $\hat{{H}}=2\sum_{{k_1,k_2, k_3, k_4,\sigma_1,\sigma_2,\sigma_3,\sigma_4}} V(k_1-k_4) \langle c_{{\sigma_1}}^\dagger(k_1) c_{{\sigma_4}}(k_4) \rangle c_{{\sigma_2}}^\dagger(k_2) c_{{\sigma_3}}(k_3) \delta_{{k_4+k_3,k_2+k_1}}$.

## Reduce momentum in Hartree term (momentum in BZ + reciprocal lattice)
**Prompt:**  
You will be instructed to simplify the Hartree term in {Hartree_second_quantized_symbol} by reducing the momentum inside the expected value {expected_value}.  
The expected value {expected_value} is only nonzero when the two momenta $k_i,k_j$ are the same, namely, {expected_value_nonzero}.  
You should use the property of Kronecker delta function $\delta_{{k_i,k_j}}$ to reduce one momentum $k_i$ but not $b_i$.
Once you reduce one momentum inside the expected value $\langle\dots\rangle$. You will also notice the total momentum conservation will reduce another momentum in the quadratic term. Therefore, you should end up with only two momenta left in the summation.  
You should follow the EXAMPLE below to reduce one momentum in the Hartree term, and another momentum in the quadratic term.  
You should recall that {Hartree_second_quantized_symbol} is {expression_Hartree}.  
Return the final simplified Hartree term {Hartree_second_quantized_symbol}.

===  
EXAMPLE:  
Given a Hamiltonian where the Hartree term $\hat{{H}}^{{Hartree}}=\sum_{{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4}} V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_4) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_3) \delta_{{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}}$, where $k_i$ is the momentum inside first Brilloun zone and $b_i$ is the reciprocal lattice.   
Inside the expected value, we realize $\langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_4) \rangle$ is nonzero only when $k_1=k_4$, i.e., $\langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_4) \rangle=\langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_4) \rangle\delta_{{k_1,k_4}}$.  
Thus, the Hartree term becomes $\sum_{{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4}} V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_4) \rangle \delta_{{k_1,k_4}} c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_3) \delta_{{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}}$.  
Use the property of Kronecker delta function $\delta_{{k_1,k_4}}$ to sum over $k_4$, we have $\sum_{{k_1, k_2, k_3,b_1,b_2,b_3,b_4}} V(k_1-k_1+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_3) \delta_{{k_1+k_2+b_1+b_2,k_3+k_1+b_3+b_4}}=\sum_{{k_1, k_2, k_3,b_1,b_2,b_3,b_4}} V(b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_3) \delta_{{k_2+b_1+b_2,k_3+b_3+b_4}}$.  
Because $k_i$ is momentum inside first Brilloun zone while $b_i$ is the reciprocal lattice. It is only when $k_2=k_3$ that $\delta_{{k_2+b_1+b_2,k_3+b_3+b_4}}$ is nonzero, i.e., $\delta_{{k_2+b_1+b_2,k_3+b_3+b_4}}=\delta_{{b_1+b_2,b_3+b_4}}\delta_{{k_2,k_3}}$. Therefore, the Hartree term simplifies to $\sum_{{k_1, k_2, k_3,b_1,b_2,b_3,b_4}} V(b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_3) \delta_{{b_1+b_2,b_3+b_4}}\delta_{{k_2,k_3}}=\sum_{{k_1, k_2,b_1,b_2,b_3,b_4}} V(b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_2) \delta_{{b_1+b_2,b_3+b_4}}$.  
Therefore, the final simplified Hartree term after reducing two momenta is $\hat{{H}}^{{Hartree}}=\sum_{{k_1, k_2,b_1,b_2,b_3,b_4}}  V(b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_4}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_3}}(k_2) \delta_{{b_1+b_2,b_3+b_4}}$ 

## Reduce momentum in Hartree term (momentum in BZ)
**Prompt:**  
You will be instructed to simplify the Hartree term, {Hartree_second_quantized_symbol}, by reducing the momentum inside the expected value {expected_value}.  
The expected value {expected_value} is only nonzero when the two momenta $k_i,k_j$ are the same, namely, {expected_value_nonzero}.  
You should use the property of Kronecker delta function $\delta_{{k_i,k_j}}$ to reduce one momentum $k_i$.
Once you reduce one momentum inside the expected value $\langle\dots\rangle$. You will also notice the total momentum conservation will reduce another momentum in the quadratic term. Therefore, you should end up with only two momenta left in the summation.  
You should follow the EXAMPLE below to reduce one momentum in the Hartree term, and another momentum in the quadratic term.  
You should recall that {Hartree_second_quantized_symbol} is {expression_Hartree}.  
Return the final simplified Hartree term {Hartree_second_quantized_symbol}.

===  
EXAMPLE:  
Given a Hamiltonian where the Hartree term $\hat{{H}}^{{Hartree}}=\sum_{{k_1,k_2, k_3, k_4,s_1,s_2}} V(k_1-k_4) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_4) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_3) \sum_{{G}}\delta_{{k_1+k_2-k_3-k_4,G}}$, where $k_i$ is the momentum inside first Brilloun zone, $G$ is the reciprocal lattice vectors, and $s_i$ is a certain index for the degree of freedom other than momentum.   
Inside the expected value, we realize $\langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_4) \rangle$ is nonzero only when $k_1=k_4$, i.e., $\langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_4) \rangle=\langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_4) \rangle\delta_{{k_1,k_4}}$.  
Thus, the Hartree term becomes $\sum_{{k_1,k_2, k_3, k_4,s_1,s_2}} V(k_1-k_4) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_4) \rangle \delta_{{k_1,k_4}} c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_3) \sum_{{G}}\delta_{{k_1+k_2-k_3-k_4,G}}$.  
Use the property of Kronecker delta function $\delta_{{k_1,k_4}}$ to sum over $k_4$, we have $\sum_{{k_1, k_2, k_3,s_1,s_2}} V(k_1-k_1) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_3) \sum_{{G}}\delta_{{k_1+k_2-k_3-k_1,G}}=\sum_{{k_1, k_2, k_3,s_1,s_2}} V(0) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_3) \sum_{{G}}\delta_{{k_2-k_3,G}}$.  
We can further simplify $\sum_{{G}}\delta_{{k_2-k_3,G}}$. Because $k_i$ is momentum inside first Brilloun zone, and the difference between $k_2$ and $k_3$ cannot exceed the first shell of reciprocal lattice vector, which means $G$ can only take the value of the origin point in the reciprocal lattice, therefore, $\sum_{{G}}\delta_{{k_2-k_3,G}}=\delta_{{k_2-k_3,0}}$.   
Thus, the Hartree term simplifies to $\sum_{{k_1, k_2, k_3,s_1,s_2}} V(0) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_3) \delta_{{k_2-k_3,0}}=\sum_{{k_1, k_2,s_1,s_2}} V(0) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_2)$.  
Therefore, the final simplified Hartree term after reducing one momentum is $\hat{{H}}^{{Hartree}}=\sum_{{k_1, k_2,s_1,s_2}} V(0) \langle c_{{s_1}}^\dagger(k_1) c_{{s_1}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_2}}(k_2)$ 


## Reduce momentum in Fock term (momentum in BZ + reciprocal lattice)
**Prompt:**  
You will be instructed to simplify the Fock term in {Fock_second_quantized_symbol} by reducing the momentum inside the expected value {expected_value}.  
The expected value {expected_value} is only nonzero when the two momenta $k_i,k_j$ are the same, namely, {expected_value_nonzero}.  
You should use the property of Kronecker delta function $\delta_{{k_i,k_j}}$ to reduce one momentum $k_i$ but not $b_i$.  
Once you reduce one momentum inside the expected value $\langle\dots\rangle$. You will also notice the total momentum conservation will reduce another momentum in the quadratic term. Therefore, you should end up with only two momenta left in the summation.
You should follow the EXAMPLE below to reduce one momentum in the Fock term, and another momentum in the quadratic term.    
You should recall that {Fock_second_quantized_symbol} is {expression_Fock}.  
Return the final simplified Fock term {Fock_second_quantized_symbol}.

===  
EXAMPLE:  
Given a Hamiltonian where the Fock term $\hat{{H}}^{{Fock}}=-\sum_{{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4}} V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_3) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_4)  \delta_{{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}}$, where $k_i$ is the momentum inside first Brilloun zone and $b_i$ is the reciprocal lattice.   
Inside the expected value, we realize $\langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_3) \rangle$ is nonzero only when $k_1=k_3$, i.e., $\langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_3) \rangle=\langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_3) \rangle\delta_{{k_1,k_3}}$.    
Thus, the Fock term becomes $-\sum_{{k_1,k_2, k_3, k_4,b_1,b_2,b_3,b_4}} V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_3) \rangle \delta_{{k_1,k_3}} c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_4) \delta_{{k_1+k_2+b_1+b_2,k_3+k_4+b_3+b_4}}$.  
Use the property of Kronecker delta function $\delta_{{k_1,k_3}}$ to sum over $k_3$, we have $-\sum_{{k_1,k_2, k_4,b_1,b_2,b_3,b_4}} V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_4) \delta_{{k_1+k_2+b_1+b_2,k_1+k_4+b_3+b_4}}=-\sum_{{k_1,k_2, k_4,b_1,b_2,b_3,b_4}} V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_4) \delta_{{k_2+b_1+b_2,k_4+b_3+b_4}}$.  
Because $k_i$ is momentum inside first Brilloun zone while $b_i$ is the reciprocal lattice. It is only when $k_2=k_4$ that $\delta_{{k_2+b_1+b_2,k_4+b_3+b_4}}$ is nonzero, i.e., $\delta_{{k_2+b_1+b_2,k_4+b_3+b_4}}=\delta_{{b_1+b_2,b_3+b_4}}\delta_{{k_2,k_4}}$.  
Therefore, the Fock term simplifies to $-\sum_{{k_1,k_2, k_4,b_1,b_2,b_3,b_4}}  V(k_1-k_4+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_4) \delta_{{b_1+b_2,b_3+b_4}}\delta_{{k_2,k_4}}=-\sum_{{k_1,k_2, b_1,b_2,b_3,b_4}} V(k_1-k_2+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_2) \delta_{{b_1+b_2,b_3+b_4}}$.  
Therefore, the final simplified Fock term after reducing two momenta is $\hat{{H}}^{{Fock}}=-\sum_{{k_1, k_2,b_1,b_2,b_3,b_4}}  V(k_1-k_2+b_1-b_4) \langle c_{{b_1}}^\dagger(k_1) c_{{b_3}}(k_1) \rangle c_{{b_2}}^\dagger(k_2) c_{{b_4}}(k_2) \delta_{{b_1+b_2,b_3+b_4}}$ 

## Reduce momentum in Fock term (momentum in BZ)
**Prompt:**  
You will be instructed to simplify the Fock term in {Fock_second_quantized_symbol} by reducing the momentum inside the expected value {expected_value}.  
The expected value {expected_value} is only nonzero when the two momenta $k_i,k_j$ are the same, namely, {expected_value_nonzero}.  
You should use the property of Kronecker delta function $\delta_{{k_i,k_j}}$ to reduce one momentum $k_i$.  
Once you reduce one momentum inside the expected value $\langle\dots\rangle$. You will also notice the total momentum conservation will reduce another momentum in the quadratic term. Therefore, you should end up with only two momenta left in the summation.
You should follow the EXAMPLE below to reduce one momentum in the Fock term, and another momentum in the quadratic term.    
You should recall that {Fock_second_quantized_symbol} is {expression_Fock}.  
Return the final simplified Fock term {Fock_second_quantized_symbol}.

===  
EXAMPLE:  
Given a Hamiltonian where the Fock term $\hat{{H}}^{{Fock}}=-\sum_{{k_1,k_2, k_3, k_4,s_1,s_2}} V(k_1-k_4) \langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_3) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_1}}(k_4)  \sum_{{G}}\delta_{{k_1+k_2-k_3-k_4,G}}$, where $k_i$ is the momentum inside first Brilloun zone, $G$ is the reciprocal lattice vectors, and $s_i$ is a certain index for the degree of freedom other than momentum.  
Inside the expected value, we realize $\langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_3) \rangle$ is nonzero only when $k_1=k_3$, i.e., $\langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_3) \rangle=\langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_3) \rangle\delta_{{k_1,k_3}}$.    
Thus, the Fock term becomes $-\sum_{{k_1,k_2, k_3, k_4,s_1,s_2}} V(k_1-k_4) \langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_3) \rangle \delta_{{k_1,k_3}} c_{{s_2}}^\dagger(k_2) c_{{s_1}}(k_4) \sum_{{G}}\delta_{{k_1+k_2-k_3-k_4,G}}$.  
Use the property of Kronecker delta function $\delta_{{k_1,k_3}}$ to sum over $k_3$, we have $-\sum_{{k_1,k_2, k_4,s_1,s_2}} V(k_1-k_4) \langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_1}}(k_4) \sum_{{G}}\delta_{{k_2-k_4,G}}$.  
We can further simplify $\sum_{{G}}\delta_{{k_2-k_4,G}}$. Because $k_i$ is momentum inside first Brilloun zone, and the difference between $k_2$ and $k_4$ cannot exceed the first shell of reciprocal lattice vector, which means $G$ can only take the value of the origin point in the reciprocal lattice, therefore, $\sum_{{G}}\delta_{{k_2-k_4,G}}=\delta_{{k_2-k_4,0}}$.   
Thus, the Fock term simplifies to $-\sum_{{k_1,k_2, k_4,s_1,s_2}} V(k_1-k_4) \langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_1}}(k_4) \delta_{{k_2-k_4,0}}=-\sum_{{k_1, k_2,s_1,s_2}} V(k_1-k_2) \langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_1}}(k_2)$.  
Therefore, the final simplified Fock term after reducing one momentum is $\hat{{H}}^{{Fock}}=-\sum_{{k_1, k_2,s_1,s_2}} V(k_1-k_2) \langle c_{{s_1}}^\dagger(k_1) c_{{s_2}}(k_1) \rangle c_{{s_2}}^\dagger(k_2) c_{{s_1}}(k_2)$ 



## Combine the Hartree and Fock term
**Prompt:**  
You will now be instructed to combine the Hartree term {Hartree_symbol} and the Fock term {Fock_symbol}.  
You should recall that the Hartree term {Hartree},  
and the Fock term {Fock}.  
You should perform the same trick of relabeling the index in the Fock term to make the quadratic operators in the Fock term the same as those in the Hartree term. The relabeling should be done with a swap : {swap_rule}.
You should add them, relabel the index in Fock term, and simply their sum. 
Return the final sum of Hartree and Fock term. 

# Mathematical Simplify
## Mathematical simplify: inner product expansion
**Prompt:**  
You will be instructed to simplify {symbol} by using $k\cdot \hat{{x}}=k_x$ and $k\cdot \hat{{y}}=k_y$ only.  
You should recall that {expression_symbol}.  
Return the simplified {symbol}. 

## Mathematical simplify: Euler's formula
**Prompt:**  
You will be instructed to simplify {symbol} by converting the exponential to the trigonometrical functions using Euler's formula only. You should also apply basic trigonometrical function simplification afterwards if possible. 
You should recall that {expression_symbol}.  
Return the simplified {symbol}.  

## Mathematical simplify: prosthaphaeresis
**Prompt:**  
You will be instructed to simplify {symbol} by reducing trigonometrical functions using prosthaphaeresis.  
You should recall that {expression_symbol}.  
Return the simplified {symbol}. 

## Mathematical simplify: Expand the product using Associative property
**Prompt:**  
You will be instructed to expand the product in {symbol}.  
You should recall that {expression_symbol}.   
Express {symbol} in the expanded form.

## Mathematical simplify: Combine using Associative property
**Prompt:**  
You will be instructed to expand the product in {symbol}.  
After expansion, you can introduce a sign variable {variable}, which take values of {values} to combine the {num_of_terms} term in to one term.
You should recall that {expressoin_symbol}.
Express the {symbol} in the expanded form.


## Mathematical simplify: reduce index
**Prompt:**  
You will be instructed to simplify the {symbol} by performing constant term summation by reducing the unnecessary indices {index}.
You should recall that {expressoin_symbol}.  
Return the simplified Kinetic Hamiltonian {symbol}.