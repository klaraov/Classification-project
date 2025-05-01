function plotConfusionMatrix(confMatrix, classLabels, titleText)
    rowSums = sum(confMatrix, 2);
    confNorm = 100 * confMatrix ./ rowSums;
    classLabels = string(classLabels(:)');

    h = heatmap(classLabels, classLabels, confNorm);
    h.CellLabelFormat = '%.1f%%';
    h.CellLabelColor = 'auto';

    h.Colormap = flipud(gray);
    h.ColorLimits = [0 100];
    h.Title = titleText;
    h.XLabel = 'Predicted Class';
    h.YLabel = 'True Class';
    h.FontSize = 13;
end
