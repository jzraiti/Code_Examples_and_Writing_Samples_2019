
classdef sigmoidGate
    %object class structure that will store these two values on every wire
    %hope this works
    properties
        ubottom
        utop
    end
    methods
        function this = sigmoidGate() %constructor method
        end
        function this = forward(this,a) % for this to work reassign obj when calling
            %eg a1 = a1.forward(unit1,2)
            this.ubottom = a;
            this.utop = Unit(1/(1+exp(-this.ubottom.value)),0.0);
              
        end
        function this = backward(this)
            %gradient must be set by upper layers: will be 0 otherwise
            s = (1/(1+exp(-this.ubottom.value)));
            this.ubottom.gradient = this.ubottom.gradient + ((s*(1-s)) * this.utop.gradient);
   
        end
    end
end


