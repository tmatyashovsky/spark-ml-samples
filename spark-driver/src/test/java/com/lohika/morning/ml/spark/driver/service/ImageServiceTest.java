package com.lohika.morning.ml.spark.driver.service;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import static org.junit.Assert.assertEquals;
import org.junit.Test;
import org.springframework.beans.factory.annotation.Autowired;


public class ImageServiceTest extends BaseTest {

    @Autowired
    private ImageService imageService;

    @Autowired
    private MLlibUtilityService utilityService;

    @Test
    public void should() throws IOException {
        imageService.processImages(getPathToTestCat(), "processed", "training.csv",false);

        Path catForTestingCsvFilePath = Paths.get(getPathToTestCat()).getParent().resolve("processed/training.csv");

        Dataset<Row> catForTestingFromCsv = getSparkSession()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", "false")
                .option("inferSchema", "true")
                .load(catForTestingCsvFilePath.toString());

        JavaRDD<LabeledPoint> catForTestingFromCsvRDD = utilityService.csvToLabeledPoint(catForTestingFromCsv);
        List<LabeledPoint> expectedResultForCatFromCsv = catForTestingFromCsvRDD.collect();

        assertEquals(1, expectedResultForCatFromCsv.size());
        assertEquals(10000, expectedResultForCatFromCsv.get(0).features().size());
        assertEquals(0, Double.valueOf(expectedResultForCatFromCsv.get(0).label()).intValue());

        Path catForTestingParquetFilePath = Paths.get(getPathToTestCat()).getParent().resolve("processed/cat.parquet");

        utilityService.labeledPointsToParquet(catForTestingCsvFilePath.toString(), catForTestingParquetFilePath.toString());

        Dataset<Row> catForTestingFromParquet = getSparkSession().read().parquet(catForTestingParquetFilePath.toString());

        JavaRDD<LabeledPoint> catForTestingFromParquetRDD = utilityService.parquetToLabeledPoint(catForTestingFromParquet);
        List<LabeledPoint> expectedResultForCatFromParquet = catForTestingFromParquetRDD.collect();

        assertEquals(1, expectedResultForCatFromParquet.size());
        assertEquals(10000, expectedResultForCatFromParquet.get(0).features().size());
        assertEquals(0, Double.valueOf(expectedResultForCatFromParquet.get(0).label()).intValue());

        assertEquals(expectedResultForCatFromCsv, expectedResultForCatFromParquet);
        assertEquals(expectedResultForCatFromCsv.get(0), expectedResultForCatFromParquet.get(0));

        // TODO: clean.
//        Files.delete(Paths.get(getPathToTestCat()).getParent());
//        Files.delete(Paths.get(getPathToTestCat()).getParent().resolve("processed"));
    }

    public String getPathToTestCat() {
        URL resource = this.getClass().getResource("/train/dog.jpg");

        return resource.getPath();
    }

}
