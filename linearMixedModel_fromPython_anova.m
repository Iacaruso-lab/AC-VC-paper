function [pVal_areas] = linearMixedModel_fromPython_anova(df_path, formula)

% df_path = 'Z:\\home\\shared\\Alex_analysis_camp\\CS_dataset_all\\analysisOutputs\\df_prop_forTest.csv';
% 
% formula = 'proportion_centre ~ area + Inj_AP + Inj_DV + (1|animal)'

df = readtable(df_path);
% df(strcmp(df.stream,'V1'),:)=[];
% df(strcmp(df.stream,'OUT'),:)=[];
model1 = fitlme(df, formula); %% area as a random effect
an = anova(model1);
for i=1:size(an,1)
    name = an{i,1};
    if contains(name,'area') || contains(name,'stream') || contains(name,'hierarchy') 
        idx = i;
        break
    end
end

% this = dataset2struct(an);
pVal_areas = an{idx,5};

% model_red = fitlme(df, 'proportion_centre ~ area'); %% area as a random effect
% compare(model_red, model1, "NSim",100) %if pval is sig, the random effects are significant
% save(outputPath, 'res');

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