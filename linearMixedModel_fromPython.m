function [res, fitLines, fitCI] = linearMixedModel_fromPython(df_path, formula,outputPath)

% df_path = 'Z:\\home\\shared\\Alex_analysis_camp\\FV_dataset_all\\analysisOutputs\\df_bySession_green_freq_forLMM.csv';
% formula = ['peakElev_bySession ~ elev + (1|animal)'];  
% % df_path = 'Z:\\home\\shared\\Alex_analysis_camp\\CS_dataset_all\\analysisOutputs\\df_forTest.csv'
% 
df = readtable(df_path);

% df(ismember(df.area,'OUT'),:)=[];
% df(ismember(df.area,'AL'),:)=[];
% df(ismember(df.area,'LM'),:)=[];
% df(ismember(df.area,'P'),:)=[];
% df(ismember(df.area,'POR'),:)=[];
% df(ismember(df.area,'LI'),:)=[];
% df(ismember(df.area,'PM'),:)=[];
% df(ismember(df.area,'AM'),:)=[];

model1 = fitlme(df, formula) %% area as a random effect
res0 = model1.Coefficients;
% anova(model1)

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

% tblnew = table();
% tblnew.y = linspace(min(df.y),max(df.y), height(df))';
% tblnew.x = zeros(height(df),1);
% % tblnew.x = linspace(min(df.x),max(df.x),height(df))';
% tblnew.animal =zeros(height(df),1);
% tblnew.sessionIdx = zeros(height(df),1);
% % [~,~,rEffects] = randomEffects(model1);
% 
% 
% [ypred,yCI,DF] = predict(model1,tblnew);
% 
% figure(); 
% h1 = line(tblnew.y,ypred);
% hold on;
% h2 = plot(tblnew.y,yCI,'g-.');hold on
% scatter(df.y, df.spline_peak)
% fitLine = res0.Estimate(1) + res0.Estimate(3)*tblnew.y
% h3 = line(tblnew.y,fitLine, color='k');
% 
% 
% figure(); 
% h1 = line(tblnew.x,ypred);
% hold on;
% h2 = plot(tblnew.x,yCI,'g-.');hold on
% scatter(df.x, df.spline_peak)














end