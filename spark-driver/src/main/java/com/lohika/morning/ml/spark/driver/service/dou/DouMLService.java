package com.lohika.morning.ml.spark.driver.service.dou;

import com.lohika.morning.ml.spark.distributed.library.function.map.dou.DouConverter;
import com.lohika.morning.ml.spark.distributed.library.function.map.dou.ProgrammingLanguage;
import com.lohika.morning.ml.spark.driver.service.MLService;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class DouMLService {

    @Autowired
    private SparkSession sparkSession;

    @Autowired
    private MLService mlService;

    public Map<String, Object> trainDOURegressionModel(String trainingSetCsvFilePath, String modelOutputDirectory) {
        Dataset<Row> trainingSet = readRawTrainingSet(trainingSetCsvFilePath);

        DouRegressionFeatureExtractor featureExtractor = new DouRegressionFeatureExtractor();

        LinearRegression linearRegression = new LinearRegression();

        Pipeline pipeline = new Pipeline().setStages(
                new PipelineStage[]{
                        featureExtractor,
                        linearRegression});

        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(linearRegression.regParam(), new double[] {0.3, 0.8})
                .addGrid(linearRegression.fitIntercept())
                .addGrid(linearRegression.elasticNetParam(), new double[] {0.3, 0.8})
                .addGrid(linearRegression.maxIter(), new int[] {50, 100})
                .build();

        // CrossValidator will try all combinations of values and determine best model using the evaluator.
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new RegressionEvaluator())
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(10);

        // Run cross-validation, and choose the best set of parameters.
        CrossValidatorModel model = crossValidator.fit(trainingSet);

        mlService.saveModel(model, modelOutputDirectory);

        return getRegressionPipelineStatistics(model);
    }

    public Double predictSalary(Double experience, String englishLevel, String programmingLanguage, String modelDirectory) {
        Dataset<Row> features = sparkSession.createDataFrame(
                Collections.singletonList(RowFactory.create(-1000, experience, englishLevel, programmingLanguage)),
                new StructType(new StructField[]{
                        DataTypes.createStructField("salary", DataTypes.IntegerType, false),
                        DataTypes.createStructField("experience", DataTypes.DoubleType, false),
                        DataTypes.createStructField("english_level", DataTypes.StringType, false),
                        DataTypes.createStructField("programming_language", DataTypes.StringType, false),
                }));

        CrossValidatorModel model = mlService.loadCrossValidationModel(modelDirectory);
        getRegressionPipelineStatistics(model);

        return model.bestModel().transform(features).first().getAs("prediction");
    }

    public Map<String, Object> clusterizeITSpecialists(String trainingSetCsvFilePath, Integer clusters, String modelOutputDirectory) {
        Dataset<Row> trainingSet = readRawTrainingSet(trainingSetCsvFilePath);

        DouClusteringFeatureExtractor featureExtractor = new DouClusteringFeatureExtractor();

        MaxAbsScaler maxAbsScaler = new MaxAbsScaler().setInputCol("features").setOutputCol("scaledFeatures");

        // Trains a k-means model.
        KMeans kmeans = new KMeans().setK(clusters).setFeaturesCol("scaledFeatures");

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[] {featureExtractor, maxAbsScaler, kmeans});

        PipelineModel pipelineModel = pipeline.fit(trainingSet);

        mlService.saveModel(pipelineModel, modelOutputDirectory);

        return getClusteringPipelineStatistics(pipelineModel);
    }

    public Integer predictLevel(Integer salary, Double experience, String englishLevel, String modelDirectory) {
        Dataset<Row> features = sparkSession.createDataFrame(
                Collections.singletonList(RowFactory.create(salary, experience, englishLevel)),
                                          new StructType(new StructField[]{
                                              DataTypes.createStructField("salary", DataTypes.IntegerType, false),
                                              DataTypes.createStructField("experience", DataTypes.DoubleType, false),
                                              DataTypes.createStructField("english_level", DataTypes.StringType, false)
                                          }));

        PipelineModel pipelineModel = mlService.loadPipelineModel(modelDirectory);
        getClusteringPipelineStatistics(pipelineModel);

        return pipelineModel.transform(features).first().getAs("prediction");
    }

    private Dataset<Row> readRawTrainingSet(String douTrainingSetCsvFilePath) {
        Dataset<Row> rawTrainingSet = sparkSession
                .read()
                .option("header", "true")
                .schema(getTrainingSetSchema())
                .csv(douTrainingSetCsvFilePath);

        rawTrainingSet.count();
        rawTrainingSet.cache();

        return rawTrainingSet;
    }

    private StructType getTrainingSetSchema() {
        return new StructType(new StructField[] {
                new StructField("id", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("city", DataTypes.StringType, true, Metadata.empty()),
                new StructField("salary", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("salary_delta", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("position", DataTypes.StringType, true, Metadata.empty()),
                new StructField("experience", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("current_job_experience", DataTypes.DoubleType, true, Metadata.empty()),
                new StructField("programming_language", DataTypes.StringType, true, Metadata.empty()),
                new StructField("specialization", DataTypes.StringType, true, Metadata.empty()),
                new StructField("age", DataTypes.IntegerType, true, Metadata.empty()),
                new StructField("sex", DataTypes.StringType, true, Metadata.empty()),
                new StructField("education", DataTypes.StringType, true, Metadata.empty()),
                new StructField("university", DataTypes.StringType, true, Metadata.empty()),
                new StructField("is_student", DataTypes.BooleanType, true, Metadata.empty()),
                new StructField("english_level", DataTypes.StringType, true, Metadata.empty()),
                new StructField("company_size", DataTypes.StringType, true, Metadata.empty()),
                new StructField("company_type", DataTypes.StringType, true, Metadata.empty()),
                new StructField("domain", DataTypes.StringType, true, Metadata.empty())
        });
    }

    private Map<String, Object> getRegressionPipelineStatistics(CrossValidatorModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Arrays.sort(model.avgMetrics());
        modelStatistics.put("Best avg metrics", model.avgMetrics()[model.avgMetrics().length - 1]);

        LinearRegressionModel linearRegressionModel = (LinearRegressionModel) ((PipelineModel) model.bestModel()).stages()[1];
        modelStatistics.put("Best intercept", linearRegressionModel.intercept());
        modelStatistics.put("Best max iterations", linearRegressionModel.getMaxIter());
        modelStatistics.put("Best reg parameter", linearRegressionModel.getRegParam());
        modelStatistics.put("Best elastic net parameter", linearRegressionModel.getElasticNetParam());

        Map<String, Object> coefficientsExplained = new HashMap<>();
        double[] coefficients = linearRegressionModel.coefficients().toArray();

        coefficientsExplained.put("Experience", coefficients[0]);
        coefficientsExplained.put("English level", coefficients[1]);

        for (int i = 2; i < coefficients.length; i++) {
            ProgrammingLanguage programmingLanguage = DouConverter.transformProgrammingLanguage(i-2);
            coefficientsExplained.put("Language " + programmingLanguage.getName(), coefficients[i]);
        }

        modelStatistics.put("Coefficients", coefficientsExplained);

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    private Map<String, Object> getClusteringPipelineStatistics(PipelineModel model) {
        Map<String, Object> modelStatistics = new HashMap<>();

        Vector[] centers = ((KMeansModel) model.stages()[2]).clusterCenters();

        for (int i = 1; i<=centers.length; i++) {
            modelStatistics.put("Cluster " + i,
                    "Salary: " + centers[i-1].apply(0) +
                            ", Experience: " + centers[i-1].apply(1) +
                            ", English level: " + centers[i-1].apply(2));
        }

        printModelStatistics(modelStatistics);

        return modelStatistics;
    }

    private void printModelStatistics(Map<String, Object> modelStatistics) {
        System.out.println("\n------------------------------------------------");
        System.out.println("Model statistics:");
        System.out.println(modelStatistics);
        System.out.println("------------------------------------------------\n");
    }

}
