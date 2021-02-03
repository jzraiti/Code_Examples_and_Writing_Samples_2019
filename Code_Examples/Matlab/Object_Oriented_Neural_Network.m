%A neural nework is essentially: an insanely complex function, 
%with random parameterization: that is nested in an optimization function 

%Objective: Build a rudimentary Neural Network from the ground up

%Part 1: Testing strategies and components

%Part 2: Support Vector Machine Construction

%Note : Most Basic "Neuron": the Real Valued Circut (function at bottom)


% ********** PART 1: ***************************************

%goal: to find the x,y values that return the highest result
x = -2;
y = 3;
disp (forwardMultiplyGate(x, y)) %returns -6. Exciting.

%simplest strategy: random local search
%change x and y by very little and keep improvements
h = 0.1;
best_out = -6;
best_x = x;
best_y = y;
for i = 1:100
    x_try = x + h * (rand * 2 - 1); % tweak x a bit
    y_try = y + h * (rand * 2 - 1); % tweak y a bit
    out = forwardMultiplyGate(x_try, y_try);
    if(out > best_out)
        %best improvement yet! Keep track of the x and y
        best_out = out; 
        best_x = x_try;
        best_y = y_try;
    end
end

fprintf('best x is %2.4f \n best y is %2.4f \n best out is %2.4f \n',best_x,best_y,best_out)


%strategy number 2: numerical gradient
%take the derivative of the function with respect to x and y respectively
%?f(x,y)/?x=(f(x+h,y)?f(x,y))/h with h being infinitely small 
%h in this case is tweak amount so ...
h = .01; %should be infinitely small but
x_derivative = (forwardMultiplyGate((x+h),y) - forwardMultiplyGate(x,y))/h;
y_derivative = (forwardMultiplyGate(x,(y+h)) - forwardMultiplyGate(x,y))/h;

fprintf('dx is %2.4f \n dy is %2.4f',x_derivative,y_derivative)% 3 and -2
%this shows the force of change with step h: x pulls result higher and y lower
%but this force is proportional to the effect hence 3 and 2
%we can use this to create a gradient change on our original variables
x = x + (h * x_derivative) % -1.97
y = y + (h * y_derivative) %2.98
disp(forwardMultiplyGate(x,y)) %-5.87 : a better result

%this is far more efficient than trying randomly: use calculus! 
%the gradient is important because it makes our steps proportional to their
%effect
%%
% Strategy 3: Analytical Gradient

%for product rules, power rules, quotient rules, etc, the derivative is
%easily defined for both x and y

%?f(x,y)?x=f(x+h,y)?f(x,y) /h=(x+h)y?xy /h=xy+hy?xy /h=hy /h=y
% no evaluations necessary 

x = -2;
y = 3;
forwardMultiplyGate(x,y) %-6
x_analgrad = y;
y_analgrad = x;

x = x + ( h * x_analgrad );
y = y + ( h * y_analgrad );

forwardMultiplyGate(x,y) % -5.87 Much simpler and faster!

%% multiple gated circuits

%can use this to optomize something simple like y = a*x + b 
%still is simple, each gate will be tweaked alone unaware of the greater
%system, its just hard to conceptualize

%try for f(x,y,z) = (x+y)z


x = -2;
y = 5;
z = -4;
disp(forwardCircuit(x,y,z)) % -12

%Lets think about analytical gradients now:
%for the multiply layer, it is still just the opposite variable
%for the addition layer, the gradient is always 1

%invoke chain rule = derivative of outside * derivative of inside
%for x (q = x+y) : ?f(q,z)/?x= (?q(x,y)/?x) * (?f(q,z)/?q)
q = forwardAddGate(x,y) %3
f = forwardMultiplyGate(q,z) %-12

%gradient of multiply gate:
fwrtz = q; %3
fwrtq = z; %-4

%derivative of add gate:
qwrtx = 1.0;
qwrty = 1.0;

%Chain rule
fwrtx = qwrtx * fwrtq; % -4
fwrty = qwrty * fwrtq; % -4 

%now change inputs accordingly
h = .01;
x = x + (h*fwrtx); %-2.04
y = y + (h*fwrty); %4.96
z = z + (h*fwrtz); %-3.97

q = forwardAddGate(x,y) %2.92
f = forwardMultiplyGate(q,z) %-11.59 improved!

%this is essentially backpropogation in a nutshell

%% Single neuron 
%for this example use: 
%f(x,y,a,b,c)=?(ax+by+c)
%with sigmoid function defined as : ?(x)=1/ 1+e?x
%with derivative of sigmoid function as : ??(x)/ ?x= ?(x)*(1??(x)) ie
%upward force per this input value x

%Note every wire has two numbers associated: 
%the value it passes forward (x) and the gradient it passes back (df(x)/dx)
%unit class will store this information

%In addition we need 3 gates: + * and sigmoid

a = Unit(1.0,0.0);
b = Unit(2.0,0.0);
c = Unit(-3.0,0.0);
x = Unit(-1.0,0.0);
y = Unit(3.0,0.0);

mulg0 = multiplyGate();
mulg1 = multiplyGate();
addg0 = addGate();
addg1 = addGate();
sg0 = sigmoidGate();
%% do forward pass
[s,mulg0,mulg1,addg0,addg1,sg0] = forwardNeuron(a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0);
%s =0.8808 = utop.value
%% Do backward pass
sg0.utop.gradient = 1.0;
[a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0] = backwardNeuron(a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0);
%% tweat inputs to make them better
step_size = 0.01;
a.value = a.value + (step_size * a.gradient);  % a.grad is -0.105
b.value = b.value + (step_size * b.gradient); % b.grad is 0.315
c.value = c.value + (step_size * c.gradient);  % c.grad is 0.105
x.value = x.value + (step_size * x.gradient);  % x.grad is 0.105
y.value = y.value + (step_size * y.gradient); % y.grad is 0.210

[s,mulg0,mulg1,addg0,addg1,sg0] = forwardNeuron(a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0);
disp(s) %8826



%**********************************************************
%% PART 2: THE REAL FUN BEGINS: Support Vector Machine

% reset the svm
svm = SVM();


x1 = 1:.5:100; %x is range 1 to 100
y1 = x1*4 - 6; %ax + by + c = 1 or -1 
labels1 = ones(1,length(x1)); %labels are 1 for everything on this line


%a = 1;
%b = 394;
%y2 = (b-a).*rand(1,length(x1)) + a;
y2 = rand(1,length(x1));



data1 = [ x1 ; y1];
data2 = [ x1 ; y2];
labels2 = ones(1,991)*-1;



for i = 1:length(x1)
    if y2(i) == y1(i)
        labels2(i) = 1;
        i
    end
    
end

labels3 = [labels1 labels2];
data3 =[data1 data2];
data = data3';
labels = labels3'; 

%% Evaluate how SVM model accuracy
for iter = 1:100
    i = randi(length(data));%change back to i when you want rand sampling
    x = data(i,1);
    y = data(i,2);
    x = Unit(x,0.0);
    y = Unit(y,0.0);
    label = labels(i);
    
    %testing svm forward function:
    %temp = svm.forward(x,y);
    %unit_out = temp.unit_out;
    %disp(unit_out)
    %Good
    %testing svm backwards:  
    %temp = temp.backward(label);
    %testing svm parameter update
    %temp = temp.parameterUpdate();
    
    svm = svm.learnFrom(x,y,label); %train the program
    
    if rem(iter,50) == 0
        disp("training accuracy")
        disp(evalTrainingAccuracy(data,labels,svm))
    end  
end



%% simplified (not modulatable) formula for optimization
a = 1;
b = -2;
c = -1;
for iter = 1:1000
    i = randi(length(data));%change back to i when you want rand sampling
    x = data(i,1);
    y = data(i,2);
    label = labels(i);
    
    score = a*x + b*y + c;
    pull = 0.0;
    if label == 1 && score < 1 %calculate pull
        pull = 1;
    end
    if label == -1 && score < -1
        pull = -1;
    end
    
    step_size = 0.01;
    a = a + (step_size * (x * (pull-a)));
    b = b + (step_size * (y * (pull-b)));
    c = c + (step_size * (1 * pull));
end

%show end coefficients
a
b
c


% ******************** FUNCTION DEFINITIONS BELOW ****************

%takes in x and y and computes x * y through gate (aka function)
function [answer] = forwardMultiplyGate(x, y)
    answer = x * y;
end
%function adds x, y 
function [answer] = forwardAddGate(x, y)
    answer = x + y;
end

%add x ,y then multiply by z
function [answer] = forwardCircuit(x, y , z)
    q = forwardAddGate(x,y); % q will represent x + y
    answer = forwardMultiplyGate(q,z);
end

%this neuron will compute the value of the 
function [s,mulg0,mulg1,addg0,addg1,sg0] = forwardNeuron(a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0)
    mulg0 = mulg0.forward(a, x); % a*x = -1
    mulg1 = mulg1.forward(b, y);% b*y = 6
    ax = mulg0.utop; %unit1 
    by = mulg1.utop; %unit2 
    
    addg0 = addg0.forward(ax,by);
    axpby = addg0.utop; % a*x + b*y = 5
    
    addg1 = addg1.forward(axpby, c);
    axpbypc = addg1.utop; % a*x + b*y + c = 2
    disp(axpbypc)
    
    sg0 = sg0.forward(axpbypc);
    s = sg0.utop; % sig(a*x + b*y + c) = 0.8808
end


%backward neuron will 
function [a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0] = backwardNeuron(a,b,c,x,y,mulg0,mulg1,addg0,addg1,sg0)
   %DO EXACT REVERSE AS BEFORE
    %must specify sg0.utop.gradient = 1.0;
    %top layer > addg1
    sg0 = sg0.backward();
    addg1.utop.gradient = sg0.ubottom.gradient;

    %addg1 >addg0 , c
    addg1 = addg1.backward(); 
    addg0.utop.gradient = addg1.Unit1.gradient;
    c.gradient = addg1.Unit2.gradient;
    disp(c.gradient) % .1050
    
    %addg0 > mulg0 mulg1
    addg0 = addg0.backward();
    mulg1.utop.gradient = addg1.Unit2.gradient;
    mulg0.utop.gradient = addg1.Unit1.gradient;
    
    mulg1 = mulg1.backward();
    mulg0 = mulg0.backward();
    
    %mulg1 > b y and mulg0 > a x
    b.gradient = mulg1.Unit1.gradient;
    y.gradient = mulg1.Unit2.gradient;
    a.gradient = mulg0.Unit1.gradient;
    x.gradient = mulg0.Unit2.gradient;
    disp(b.gradient)
    disp(y.gradient)
    disp(a.gradient)
    disp(x.gradient)

end


%Function that computes classification accuracy
function percentcorrect = evalTrainingAccuracy(data,labels,svm)
    num_correct = 0;
    for i = 1:length(data)
        x = Unit(data(i,1),0.0);
        y = Unit(data(i,2),0.0);
        true_label = labels(i);
        
        svm1 = svm.forward(x,y);
        if (svm1.unit_out.value > 0)
            predicted_label = 1;
        else
            predicted_label = -1;
        end
        %checks prediction vs label
        if(predicted_label == true_label)
            num_correct= num_correct + 1;
        end
    end
    percentcorrect = num_correct/length(data);
end

%function for learning aka tweaking the 



