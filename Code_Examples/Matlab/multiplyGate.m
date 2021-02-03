
classdef multiplyGate
    %create multiply gate
    properties
        Unit1= Unit(0.0,0.0);
        Unit2= Unit(0.0,0.0);
        utop= Unit(0.0,0.0);
    end
    methods
        function this = forward(this,a,b) % for this to work reassign obj when calling
            %eg a1 = a1.forward(unit1,2)
            this.Unit1= a;
            this.Unit2= b;
            %first arg is classinstance, other are inputs
            %utop = Unit(input1*input2,0.0);
            this.utop = Unit((this.Unit1.value * this.Unit2.value),0.0);
              
        end
        function this = backward(this)
            %gradient are the opposite units value * filled in gradient
            %from utop
            %+= because it allows us to use the output of one gate mult
            %times like wires branching out, gradients of different
            %branches just add up when computing final gradient with
            %respect to the total circuit output
            this.Unit1.gradient = this.Unit1.gradient + (this.Unit2.value  * this.utop.gradient);
            this.Unit2.gradient = this.Unit2.gradient + (this.Unit1.value  * this.utop.gradient);
        end
    end
end

