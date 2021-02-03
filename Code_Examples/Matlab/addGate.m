

classdef addGate
    %object class structure that will be addition gate
    %hope this works
    properties
        Unit1 = Unit(0.0,0.0);
        Unit2 = Unit(0.0,0.0);
        utop = Unit(0.0,0.0);
    end
    methods
        function this = forward(this,a,b) % for this to work reassign obj when calling
            %eg a1 = a1.forward(unit1,2)
            this.Unit1= a;
            this.Unit2= b;
            %first arg is classinstance, other are inputs
            %utop = Unit(input1*input2,0.0);
            this.utop = Unit((this.Unit1.value + this.Unit2.value),0.0);
        end
        function this = backward(this)
            %gradient must be set by upper layers: will be 0 otherwise
            this.Unit1.gradient = this.Unit1.gradient + (1 * this.utop.gradient);
            this.Unit2.gradient = this.Unit2.gradient + (1 * this.utop.gradient);
        end
    end
end



