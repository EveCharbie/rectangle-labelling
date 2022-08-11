for p=1:10
    subplot(2,5,p) ; imagesc(squeeze(ftAllFiles{1,p}.powspctrm))
end