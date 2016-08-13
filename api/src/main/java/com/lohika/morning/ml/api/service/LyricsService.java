package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.lyrics.TextService;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class LyricsService {

    @Value("${lyrics.training.set.directory.path}")
    private String lyricsTrainingSetDirectoryPath;

    @Value("${lyrics.model.directory.path}")
    private String lyricsModelDirectoryPath;

    @Autowired
    private TextService textService;

    public Map<String, Object> classifyLyrics() {
        return textService.classifyLyricsWithPipeline(
            lyricsTrainingSetDirectoryPath, lyricsModelDirectoryPath);
    }

    public Genre predictGenre(String unknownLyrics) {
        double prediction = textService.predict(unknownLyrics, lyricsModelDirectoryPath);

        if (0.0D == prediction) {
            return Genre.METAL;
        }

        if (1.0D == prediction) {
            return Genre.POP;
        }

        return Genre.UNKNOWN;
    }

}
