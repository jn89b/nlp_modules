% function result = pend(duration, targetangle, N, oldresult);
N = 30
duration = 3.0; 
targetangle = 180;
oldresult = 100;
	% Finds optimal motion of a torque-driven pendulum.  Task is to move from one
	% static posture to another in a given time.
	% Method: direct collocation with 3-point discretization formula for acceleration.
	
	% Author: Ton van den Bogert <a.vandenbogert@csuohio.edu>
	
	% This work is licensed under a Creative Commons Attribution 3.0 Unported License.
	% http://creativecommons.org/licenses/by/3.0/deed.en_US
	
	% Inputs:
	%	duration		duration of the movement (s)
	%	targetangle		target angle (deg)
	%	N				number of collocation nodes to use
	% 	oldresult		(optional) initial guess
	
	% Notes:
	% 1. This code may be useful as a template for solving other optimal control problems, such
	%    as cart-pole upswing.
	% 2. IPOPT will be used when it is installed, otherwise Newton's method.  IPOPT is recommended
	%    because it is more robust.  Newton's method can still solve most problems, but you may
	%    need to solve a sequence of problems of increasing difficulty to ensure convergence.
	% 3. The solution may be a local optimum, especially for tasks that involve multiple
	%    revolutions.  Try different initial guesses (maybe random) to check for this.
	
	% The following examples all converge with IPOPT or Newton:
	%	r = pend(1.0, 180, 100);		% swing up in 1 second, 100 collocation nodes
	%   r = pend(3.0, 180, 100);		% now do it in 3 seconds, note the countermovement
	%   r = pend(10.0, 180, 100);		% now do it in 10 seconds, multiple countermovements are seen
	%   r = pend(5.0, 720, 300);		% do two full revolutions in 5 seconds, 300 collocation nodes
	%   r2 = pend(..,..,..,r);			% use previous result r as initial guess
	
	% settings
% 	MaxIterations = 1000;
% 	if exist('ipopt') == 3
% 		method = 'ipopt';
% 	else
% 		disp('IPOPT is not installed.');
% 		disp('Newton method will be used, may be more sensitive to initial guess.');
% 		disp('Hit ENTER to continue...');
% 		pause
% 		method = 'newton';
% 	end
    method = 'newton'


	% initializations
	close all
	tic
	h = duration/(N-1);		% time interval between nodes
	times = h*(0:N-1)';		% list of time points
	Ncon = N-2 + 4;			% N-2 dynamics constraints and 4 task constraints

	% model parameters
	L = 1;			%	length of pendulum (m)
	m = 1;			%	mass of pendulum (kg)
	I = 1;			% 	moment of inertia relative to pivot (kg m^2)
	g = 9.81;		%	gravity (m s^-2)

	% state variable is x: angle relative to hanging down
	% control variable is u: torque applied at joint

% 	% if oldresult was provided, use it as initial guess, otherwise use zero initial guess (pendulum hangs down, no torque)
% 	if (nargin == 4)
% 		oldN = numel(oldresult.t);
% 		oldreltime = (0:oldN-1)'/(oldN-1);			% sample times of old result, from 0 to 1
% 		newreltime = (0:N-1)'/(N-1);				% sample times for new optimization, from 0 to 1
% 		x = interp1(oldreltime, oldresult.x, newreltime);
% 		u = interp1(oldreltime, oldresult.u, newreltime);
% 	else
% 		x = randn(N,1);
% 		u = randn(N,1);
% 	end

	x = randn(N,1);
	u = randn(N,1);
	% encode initial guess of unknowns into a long column vector X
	X0 = [x ; u];
	ix = (1:N);				% index to elements in X where angles x are stored
	iu = N + (1:N);			% index to elements in X where controls u are stored
	NX = size(X0,1);		% number of unknowns
	show(X0, confun(X0, Ncon, N, h));	% show the initial guess
	drawnow
	
	% X0 = X0 + 0.001*randn(size(X0));		% perturb the initial guess a little before optimizing
	
	if strcmp(method,'ipopt')
		% solve the NLP with IPOPT
		funcs.objective = @objfun;
		funcs.gradient  = @objgrad;
		funcs.constraints = @confun;
		funcs.jacobian    = @conjac;
		funcs.jacobianstructure = @conjacstructure;
		options.cl = zeros(Ncon,1);
		options.cu = zeros(Ncon,1);	
		options.ipopt.max_iter = MaxIterations;
		options.ipopt.hessian_approximation = 'limited-memory';
		[X, info] = ipopt(X0,funcs,options);
	elseif strcmp(method, 'newton')
		% solve the NLP using Newton iteration on the KKT conditions
		X = X0;
		ctol = 1e-4;		% constraint tolerance
		ftol = 1e-4;		% cost function tolerance
		xtol = 1e-4;		% solution tolerance
		F = 1e10;
		for iter=1:MaxIterations
			Fprev = F;

			% evaluate objective function F and constraint violations c
			F = objfun(X);
			G = objgrad(X);
			H = objhess(X);
			c = confun(X);
			J = conjac(X);

			% form the linearized KKT system K*x = b				
			K = [H J'; J sparse(Ncon,Ncon)];		
			b = [-G; -c];
			
			% solve the linear system K*dZ=b 
			% Z is a vector containing the unknowns X and the Lagrange multipliers.  dZ is the change in this iteration
			dZ = K\b;	
			dX = dZ(1:NX);					% the first NX are the elements of X
			
			% do a half Newton step (converges slower than full Newton step, but more likely to converge)
			% for more robust convergence, we should do a line search here to make sure we always have progress 
			X = X + dX/2;
			rmsC = sqrt(mean(c.^2));
			rmsdX = sqrt(mean(dX.^2));
			fprintf('Iter: %3d  F=%10.5e  rms(c)=%10.5e   rms(dX)=%10.5e\n', iter,F,rmsC,rmsdX);
					
			if (max(abs(c)) < ctol) && (abs(F-Fprev)<ftol) && (mean(abs(dX))<xtol)
				break;
			end 
		end
		if iter >= MaxIterations
			disp('Maximum number of iterations exceeded.');
		else
			disp('Optimal solution found');
		end
	else
		error('method not recognized');
	end
	
	% plot results
	show(X, confun(X));
	
	% make movie of the solution
	disp('Hit ENTER to generate animation...');
	pause
	fps = N/duration;
    avi = VideoWriter('pend.avi');
    open(avi);
	figure(2);
	clf;
	set(gcf,'Position',[5 100 650 650]);
	set(gcf, 'color', 'white');
	s = 1.5*L;
	for i=1:N
		plot([-s s],[0 0],'k','LineWidth',2);
		hold on
		plot([0 L*cos(X(i)-pi/2)], [0 L*sin(X(i)-pi/2)],'b-o','LineWidth',2);
		axis('equal');
		axis('square');
		axis([-s s -s s]);
		title(['t = ' num2str(times(i),'%8.3f')]);
		if (i==1)
			F = getframe(gca);
			frame = [1 1 size(F.cdata,2) size(F.cdata,1)];
		else
			F = getframe(gca,frame);
		end
		writeVideo(avi,F);
		drawnow;
		hold off;
	end
	close(avi);
%	close(2);
	
	% store results
	result.t = times;
	result.x = X(1:N);
	result.u = X(N+(1:N));
	
	% start of embedded functions
	
	%=========================================================
	function F = objfun(X);
		% objective function: integral of squared controls
		F = h * sum(X(iu).^2);
	end

	%=========================================================
	function G = objgrad(X);
		% gradient of the objective function coded in objfun
		G = zeros(NX,1);
		G(iu) = 2 * h * X(iu);
	end

	%=========================================================
	function H = objhess(X);
		% hessian of objective function coded in objfun
		H = spalloc(NX,NX,N);
		H(iu,iu) = 2 * h * speye(N,N);
	end

	%=========================================================
    function c = confun(X, Ncon, N, h)
        duration = 3.0; 
        targetangle = 180;
        oldresult = 100;

		% constraint function (dynamics constraints and task constraints)
	    L = 1;			%	length of pendulum (m)
	    m = 1;			%	mass of pendulum (kg)
	    I = 1;			% 	moment of inertia relative to pivot (kg m^2)
	    g = 9.81;		%	gravity (m s^-2)


		% size of constraint vector
		c = zeros(Ncon,1);

		% dynamics constraints
		% Note: torques at node 1 and node N do not affect movement and will therefore
		% always be zero in a minimal-effort solution.
		for i=1:N-2
			x1 = X(i); 
			x2 = X(i+1);
			x3 = X(i+2);
			u2 = X(N+i+1);
			xdd = (x3 - 2*x2 + x1)/h^2;							% three-point formula for angular acceleration
			c(i) =  xdd - ( -m * g * L*sin(x2) + u2) / I;		% equation of motion must be satisfied
		end
		
		% task constraints

		% initial position must be zero:
		c(N-1) 	= X(1);					
		% initial velocity must be zero:
		c(N) 	= X(2)-X(1);			
		% final position must be at target angle:
		c(N+1) 	= X(N) - targetangle*pi/180;	
		% final velocity must be zero:
		c(N+2) 	= X(N)-X(N-1);		

		% show current iterate, every 0.1 second
%		if toc > 0.1
			show(X,c);
			tic;
%		end
				
	end
	%=========================================================
	function J = conjac(X)
		
		% size of Jacobian
		J = spalloc(Ncon,NX,4*(N-2) + 6);

		% dynamics constraints
		for i=1:N-2
			% Jacobian matrix: derivatives of c(i) with respect to the elements of X
			x2 = X(i+1);
			J(i,i) 		= 1/h^2;
			J(i,i+1) 	= -2/h^2 + m * g * L * cos(x2) / I;
			J(i,i+2) 	= 1/h^2;
			J(i,N+i+1) 	= -1/I;
			
		end
		
		% task constraints

		% initial position must be zero:
		J(N-1,	1) = 1;
		% initial velocity must be zero:
		J(N,2) = 1;
		J(N,1) = -1;
		% final position must be at target angle:
		J(N+1,N) = 1;
		% final velocity must be zero:
		J(N+2,N) = 1;
		J(N+2,N-1) = -1;
		
	end
	%=========================================================
	function J = conjacstructure(X)
		
		% size of Jacobian
		J = spalloc(Ncon,NX,4*(N-2) + 6);

		% dynamics constraints
		for i=1:N-2
			% Jacobian matrix: derivatives of c(i) with respect to the elements of X
			J(i,i) 		= 1;
			J(i,i+1) 	= 1;
			J(i,i+2) 	= 1;
			J(i,N+i+1) 	= 1;
		end
		
		% task constraints

		% initial position must be zero:
		J(N-1,	1) = 1;
		% initial velocity must be zero:
		J(N,2) = 1;
		J(N,1) = 1;
		% final position must be at target angle:
		J(N+1,N) = 1;
		% final velocity must be zero:
		J(N+2,N) = 1;
		J(N+2,N-1) = 1;
		
	end
	%============================================================
	function show(X,c)
		% plot the current solution
		x = X(ix);
		u = X(iu);
		figure(1)
		subplot(3,1,1);plot(times,x*180/pi);title('angle')
		subplot(3,1,2);plot(times,u);title('torque');
		subplot(3,1,3);plot(c);title('constraint violations');
	end

