function [PairsMatrix] = getSuperPixelPairs(motion,color,location)

N = size(motion,2);
K = size(motion{1},2);

PairsMatrix = zeros(K,N);

for i=1:N
    for l=1:K
        lColor = color{i}{l};
        lMotion = motion{i}{l};
        lLocation = location{i}{l};
        minValue = inf;
        for r=1:K
        	rColor = color{i}{r};
            rMotion = motion{i}{r};
            rLocation = location{i}{r};
            LocationDist = sum(abs(rLocation-lLocation));
            if(LocationDist<0.12 && l~=r)%在一个局部范围内，寻找运动及颜色最相近的“一批”点
                MotionDist = sum(abs(lMotion-rMotion));
                ColorDist = sum(abs(lColor-rColor));
                if(minValue>(ColorDist))
                   %minValue = (MotionDist+ColorDist);
                   minValue = (ColorDist);
                   PairsMatrix(l,i) = r;
                end
            end
        end
    end
end