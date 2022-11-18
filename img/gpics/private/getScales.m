function [scales, npic, avgA] = getScales(smoothedPics, nscales, plotFlag)

    for k = 1:numel(smoothedPics)
    bitm = double(smoothedPics{k});
    [pyr,pind] = buildLpyr(bitm,nscales); 

        for j = 1:nscales
        npic{k, j} = reconLpyr(pyr,pind,[j]); % skala 6 ist lowpass-residual
        avg(k,j,:)= sfPlot(npic{k, j},0);
        end
        % % see https://de.wikipedia.org/wiki/Bildpyramide
        % % Laplacian Pyramids are bandpass filter, the resulting pictures have non-overlapping 
        % % spatial frequency bands
    end

% optionally: plot
avgA = squeeze(mean(avg,1));
if plotFlag
    figure; loglog(1:size(avgA,2), avgA(1:nscales,:)); xlim([0 200]);
end

erg = NaN(1, nscales-1); 
for mm = 1:(nscales-1)
erg(mm) = find(log(avgA(nscales-mm,:)) >= log(avgA(nscales-mm+1,:)),1);
end
dm   = size(smoothedPics{1});
fnam = ['LaplaceScales_', num2str(dm(1)), 'x', num2str(dm(2)), '.txt'];
fhandle = fopen(fnam, 'w');
fprintf(fhandle, '\n');
fprintf(fhandle, 'Scale 0 is 0 to %.1f cycles per face\n', erg(1));
for nn = 1:(numel(erg)-1)
fprintf(fhandle, 'Scale %u is %.1f to %.1f cycles per face\n', nn, erg(nn), erg(nn+1));
end
fprintf(fhandle, 'Scale %u is %.1f to %.1f cycles per face\n\n', numel(erg), erg(nn+1), 2*erg(nn+1));

fprintf(fhandle, '\nAlternatively, reconstructed from scale with highest resolution:');
fprintf(fhandle, '\nScale 0 is 0 to %.3f cycles per face', erg(end)/2^(numel(erg)-1));
for l = 1:(numel(erg)-1)
fprintf(fhandle, '\nScale %u is %.3f to %.3f  cycles per face', l, erg(end)/2^(numel(erg)-l), erg(end)/2^(numel(erg)-1-l));
end
fprintf(fhandle, '\nScale %u is %.3f to %.3f cycles per face\n\n', numel(erg), erg(end)/2^(numel(erg)-1-l), 2*erg(end)/2^(numel(erg)-1-l));

scales = fliplr([2*erg(end)./2^0, erg(end)./(2.^[0:4])]); % return value
fclose(fhandle);
end