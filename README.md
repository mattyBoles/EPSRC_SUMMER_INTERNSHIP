Preservation of long-term behaviour in data-driven modelling of dynamical systems with machine learning techniques.

Data-driven prediction of chaotic systems using machine learning techniques is an ever-changing research field, import for improving accuracy of forecasting in areas such as weather prediction. However, the mechanisms determining whether a network preserves chaos remain systematically unanalysed. This is the objective of this project, using Jacobian and singular value decomposition to compare a variety of neural networks on the Lorenz63 equations.

Chaos is aperiodic long-term behavior in a deterministic system that exhibits 
sensitive dependence on initial conditions.

Chaos is refers to a system in which infinitesimal perturbations grow exponentially. As a result, we are not concerned with the long-term trajectories reproduced by the model to be close to those of the true system, as even the tiniest error will result in a completely different trajectory. We are instead interested in the long-term shape and statistics of the system - whether the model learns the true dynamics of the system.

Throughout this project, we use the 4th order Runge-Kutta as a ground truth of the true Lorenz equations, with a timestep of 0.01. The project focusses on small models as, as well as being easier to analyse, they are simply more interesting.

A successful model will match the Lorenz equations in 3 areas:
- Long-term separation rates can be diagnosed with the Lyapunov Spectrum. The Lyapunov spectra is an ordered set of exponents describing the average rate of separation of infinitesimally close trajectories along a particular direction. lambda_i > 0 shows expansion, lambda_i = 0 shows no average growth or decay. lambda_i < 0 shows contraction. For chaos, we require at least one positive exponent. The true Lyapunov spectrum of the Lorenz system is 0.905, ~0, -14.5.

-Instantaneous geometry. Throughout a trajectory, we require the model to show the same local geometry as that of the true system. We analyse this via Singular Value Decomposition of the Jacobian. This shows the exponential rate of stretching and contraction, calculated at a point, with corresponding singular vectors describing the directions of maximum and minimum orthogonal growth. In the true system, SV1 and 3 have an almost sinusoidal structure, with SV1 fluctauing between 1 and 1.1, and amplitude growing with radius. Sv3 fluctuates between 0.75 and 0.9, with amplitude similarly growing with radius. Sv2, the neutral direction, fluctuates between 0.98 and 1.01 with very small changes in amplitude.

-Overall shape of the attractor. A tendency of Neural Networks is to learn the typical trajectories well, but fail to learn rare or extreme conditions. The Lorenz Attractor spends a lot of time on wide radii of the unstable points, but does find loops of smaller radii with consistency. This is a feature that must be preserved for a successful imitation of the characteristics of the Lorenz Attractor. We analyse this by plotting a histogram of the distance from the unstable points.

'''


Training hyperparameters.
We train the model using the MSE loss on a 1-step integration of the rk4 method. This is the typical method used. MSE for small networks falls into a typical hole mentioned above. The model has a tendency to learn the 'typical trajectory', resulting in a model which produces attractors in which distance from the unstable points remains too stable in comparison to the true system. WIth small widths, the loss is minimised with this typical trajectory. This is more troublesome in chaotic networks. Indeed, 2 input vectors x1 and x2 may be close together, but their outputs, f(x1) and f(x2) may be far apart. The model lacksthe capacity to learn this difference, and so learns the average of the difference. At higher widths, this becomes less of an issue. The model does tend to fall into a standard trajectory, but it realises it is far off tyrajecotry and we see a sharp drop in sv3 to correct this mistake and fix the course. We can test this as follows:
We integrate the true Lorenz system until we are confident the system is on the attractor. We integrate the true system one step, and find the dingualr values. We forward pass the inout to the model, and there too find the singular values. Then, we repeat, with the next step being th eoutput of the true system. Inthis way, we see tne true difference between singular values of the true system and model, without the worry of obscured information by diverging trajecotires.
Below we can see that the 8-wide model consistently has SV3 lower than the true system. The same can be said for width 16, with sharp overcorrection points, This manifests as Lyapunov-3 being consistently lower than -14.5, the the true Lyapunnov3.

Further, we compare directly 2 similar weight decay/ no weight decay models. We can back this up matehemtically. Explain why this is applicable more generally in small models but speciifally in chaos as it adds up to the lack of location-specific activations. Maybe we can plot sv_i over trajectory compared with real and non-weigth decayed. Here we can talk about kl-loss.

Next, activation function - relu - jacobian and v3 analysis, somehow show that true lroenz is constantly rotating, I guess this is v3, but rleu cannot -why
tanh - steep drop. Show neuron 13, deactivating, affect on sv3, and possibly direction/alignment. Explain via jacobian.
This justifies arctan and softplus, and further looking graidnets of derivitives of activatiomn functions. -> wider search?
Leanrt betas/alphas?

Initisialistaion - requires more work, is it problems with the training set up? Idk.
Similarly for trajecoties. We require a stidy on n_samples and overfitting - comes back to distirbutioins.

All this justifies 'can use MLPs - thyere markovian - but RNNs reduce the fuss.



