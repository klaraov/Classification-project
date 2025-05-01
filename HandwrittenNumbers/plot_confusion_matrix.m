function plot_confusion_matrix(matrix, labels, plot_title)
    % Normalize each row to percentages
    row_sums   = sum(matrix, 2);
    norm_matrix = 100 * matrix ./ row_sums;
    
    % Ensure labels are a row of strings
    labels = string(labels(:)');
    
    % Create heatmap
    h = heatmap(labels, labels, norm_matrix);
    h.CellLabelFormat = '%.1f%%';
    h.CellLabelColor  = 'auto';
    
    % Styling
    h.Colormap    = flipud(gray);
    h.ColorLimits = [0 100];
    h.Title       = plot_title;
    h.XLabel      = 'Predicted Class';
    h.YLabel      = 'True Class';
    h.FontSize    = 13;
end
