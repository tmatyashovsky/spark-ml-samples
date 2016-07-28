package com.lohika.morning.ml.api.service;

import com.lohika.morning.ml.spark.driver.service.TextService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class LyricsService {

    @Value("${lyrics.training.set.txt.file.path}")
    private String douTrainingSetParquetVectorFilePath;

    @Autowired
    private TextService textService;

    public void classifyLyrics() {
        textService.classifyDarkLyrics(douTrainingSetParquetVectorFilePath);
    }

}
