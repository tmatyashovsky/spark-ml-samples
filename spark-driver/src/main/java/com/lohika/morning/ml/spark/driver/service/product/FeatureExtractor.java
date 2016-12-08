package com.lohika.morning.ml.spark.driver.service.product;

import com.lohika.morning.ml.spark.distributed.library.function.map.product.ToTrainingExample;
import java.io.IOException;
import java.util.UUID;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class FeatureExtractor extends Transformer implements MLWritable {

    private String uid;

    public FeatureExtractor(String uid) {
        this.uid = uid;
    }

    public FeatureExtractor() {
        this.uid = "FeatureExtractor" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public Dataset<Row> transform(Dataset dataset) {
        dataset = dataset.map(new ToTrainingExample(),
                RowEncoder.apply(transformSchema(dataset.schema())));

//        dataset.count();
//        dataset.cache();

        return dataset;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                DataTypes.createStructField("label", DataTypes.StringType, false),
                DataTypes.createStructField("features", new VectorUDT(), false)});
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<FeatureExtractor> read() {
        return new DefaultParamsReader<>();
    }
}
