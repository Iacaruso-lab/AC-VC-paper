function [res, fitLines, fitCI] = linearMixedModel_fromPython(df_path, formula,outputPath)

df = readtable(df_path);

model1 = fitlme(df, formula) 
res0 = model1.Coefficients;

for i=1:height(res0)
    name = res0.Name{i};
    if contains(name, 'Intercept')
        name = 'Intercept';
    end
    estimate = res0.Estimate(i);
    lower = res0.Lower(i);
    upper = res0.Upper(i);
    pVal = res0.pValue(i);
    k = [estimate,pVal, lower, upper];
    eval(sprintf('res.%s = k;',name));%add the flattened variable onto a struct

    %get confidence interval
    if ~contains(name, 'Intercept')
        tblnew = table();
        for n=1:length(model1.PredictorNames)
            name0 = model1.PredictorNames{n};
            if strcmp(name, name0)
                tblnew.(name0) = linspace(0,max(df.(name)), 100)';
            else
                tblnew.(name0) = zeros(100,1);
            end

        end
       
        [ypred,yCI,DF] = predict(model1,tblnew);
        eval(sprintf('fitLines.%s = ypred;',name));%add the flattened variable onto a struct
        eval(sprintf('fitCI.%s = yCI;',name));%add the flattened variable onto a struct
    end

end

save(outputPath, 'res');















end