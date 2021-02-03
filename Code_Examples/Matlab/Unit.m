classdef Unit
    %object class structure that will store these two values on every wire
    properties
        value
        gradient
    end
    methods
        function obj = Unit(a,b) %constructor method
            obj.value = a;
            obj.gradient = b;
        end
    end
end