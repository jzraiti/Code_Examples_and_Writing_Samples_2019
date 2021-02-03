classdef Circuit
    %an arrangement of all of the gates to optimize ax + by + c
    properties
        mulg0 = multiplyGate();
        mulg1 = multiplyGate();
        addg0 = addGate();
        addg1 = addGate();
        a_gradient;
        b_gradient;
        c_gradient;
        
        axpbypc = Unit(0.0,0.0); % to store final value %has struct (value,gradient)
    end
    methods

        function this = forward(this,x,y,a,b,c) % for this to work reassign obj when calling
            this.mulg0 = this.mulg0.forward(a, x); % a*x
            ax = this.mulg0.utop;
            this.mulg1 = this.mulg1.forward(b, y); % b*y
            by = this.mulg1.utop;
            this.addg0 = this.addg0.forward(ax, by);%a*x + b*y
            axpby = this.addg0.utop;
            this.addg1 = this.addg1.forward(axpby, c); % a*x + b*y + c
            this.axpbypc = this.addg1.utop;

            %return this.axpbypc;
        end
        function this = backward(this,gradient_top)
            %ax + by + c 
            topgradient = Unit(0.0,0.0);
            topgradient.gradient = gradient_top;
            this.addg1.utop.gradient = topgradient.gradient;
            
            this.addg1 = this.addg1.backward(); % sets gradient in axpby and c
            %TOP LAYER : + c 
            this.c_gradient = this.addg1.Unit2.gradient;

            this.addg0.utop = this.addg1.Unit1; 
            this.addg0 = this.addg0.backward();
            %unit 1 ax aka mulg0 , unit 2 by aka mulg1
            
            this.mulg0.utop.gradient = this.addg0.Unit1.gradient;
            this.mulg1.utop.gradient = this.addg0.Unit2.gradient;

            %BOTTOM LAYER:a x b y
            
            this.mulg1 = this.mulg1.backward(); % sets gradient in b unit1 and y unit2
            this.a_gradient = this.mulg1.Unit1.gradient;
                     
            x = this.mulg1.Unit2.gradient;

            
            this.mulg0 = this.mulg0.backward(); % sets gradient in a unit1 and x unit2
            this.b_gradient = this.mulg0.Unit1.gradient;
            
            y = this.mulg0.Unit2.gradient;
            disp("gradients")
            disp(this.a_gradient)
            disp(this.b_gradient)
            disp(this.c_gradient)
        end
    end
end

