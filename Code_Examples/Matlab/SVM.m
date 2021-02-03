
classdef SVM
    %create multiply gate
    properties
        a = Unit(1.0,0.0);
        b = Unit(-2.0,0.0);
        c = Unit(-1.0,0.0);
        circuit = Circuit();
        
        unit_out = Unit(0.0,0.0); % to store final value
        
    end
    
    methods
        function this = SVM() %constructor method
        end

        function this = forward(this,x,y) % for this to work reassign obj when calling
            %pass back unit_out, the end result + or - value, 0.0 gradient
            %ORIGINAL %this.unit_out = this.circuit.forward(x, y, this.a, this.b, this.c);
            %return this.unit_out;
            this.circuit = this.circuit.forward(x, y, this.a, this.b, this.c);
            this.unit_out = this.circuit.axpbypc;
        end
        function this = backward(this,label) %each x,y is labeled 1 or -1
            % reset pulls on a,b,c
            
            %disp(label)
            %disp(this.a.value)
            %disp("hW")
            %disp(this.unit_out.value)
            %disp(
            this.a.gradient = 0.0; 
            this.b.gradient = 0.0; 
            this.c.gradient = 0.0;
            
            % compute the pull based on what the circuit output was
            pull = 0.0;
            
            %disp(this.unit_out.value)
            predicted_label = this.unit_out.value;
            if(label == 1 && predicted_label < 1) 
                pull = 1; %the score was too low: pull up
            end
            if(label == -1 && predicted_label > -1)
                pull = -1; % the score was too high for a positive example, pull down
            end

            this.circuit = this.circuit.backward(pull); % writes gradient into x,y,a,b,c
            this.a.gradient = this.circuit.a_gradient;
            this.b.gradient = this.circuit.b_gradient;
            this.c.gradient = this.circuit.c_gradient;
            
            % optional : add regularization pull for parameters: towards zero and proportional to value
            %this.a.gradient = this.a.gradient - this.a.value;
            %this.b.gradient = this.b.gradient - this.b.value;
            disp("final gradients")
            disp(this.a.gradient)
            disp(this.b.gradient)
            disp(this.c.gradient)
        end
        
        %activation function call
        function this = learnFrom(this,x,y,label)
            this = this.forward(x, y); % forward pass (set .value in all Units)
            this = this.backward(label); %backward pass (set .grad in all Units)
            this = this.parameterUpdate(); % parameters respond to tug
        end
        
        %Update Parameters
        function this = parameterUpdate(this)
            step_size = 0.01;

            this.a.value = this.a.value + ( step_size * this.a.gradient);
            this.b.value = this.b.value + ( step_size * this.b.gradient);
            this.c.value = this.c.value + ( step_size * this.c.gradient);
            fprintf("\n new a %2.2f , new b %2.2f , new c %2.2f \n", this.a.value, this.b.value, this.c.value)
            
        end
    end
end


