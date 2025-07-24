function [pVal_areas] = linearMixedModel_fromPython_anova(df_path, formula)

df = readtable(df_path);

model1 = fitlme(df, formula); %% area as a random effect
an = anova(model1);
for i=1:size(an,1)
    name = an{i,1};
    if contains(name,'area') || contains(name,'stream') || contains(name,'hierarchy') 
        idx = i;
        break
    end
end

pVal_areas = an{idx,5};

end