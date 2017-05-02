package com.lohika.morning.ml.spark.driver;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;

@Configuration
@PropertySource("classpath:spark.properties")
@ComponentScan("com.lohika.morning.ml.spark.driver.*")
public class SparkContextConfiguration {

    @Bean
    public SparkSession sparkSession() {
        SparkConf sparkConf = new SparkConf()
                .setMaster(master)
                .setAppName(applicationName)
                .setJars(distributedLibraries)
                .set("spark.cores.max", coresMax)
                .set("spark.driver.memory", driverMemory)
                .set("spark.executor.memory", executorMemory)
                .set("spark.serializer", serializer)
                .set("spark.kryoserializer.buffer.max", kryoserializerBufferMax)
                .set("spark.kryo.registrationRequired", "false")
                .set("spark.sql.shuffle.partitions", sqlShufflePartitions)
                .set("spark.default.parallelism", defaultParallelism);

        SparkContext sparkContext = new SparkContext(sparkConf);

        return new SparkSession(sparkContext);
    }

    @Value("${spark.master}")
    private String master;

    @Value("${spark.application-name}")
    private String applicationName;

    @Value("${spark.distributed-libraries}")
    private String[] distributedLibraries;

    @Value("${spark.cores.max}")
    private String coresMax;

    @Value("${spark.driver.memory}")
    private String driverMemory;

    @Value("${spark.executor.memory}")
    private String executorMemory;

    @Value("${spark.serializer}")
    private String serializer;

    @Value("${spark.sql.shuffle.partitions}")
    private String sqlShufflePartitions;

    @Value("${spark.default.parallelism}")
    private String defaultParallelism;

    @Value("${spark.kryoserializer.buffer.max}")
    private String kryoserializerBufferMax;

    @Bean
    // Static is extremely important here.
    // It should be created before @Configuration as it is also component.
    public static PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer() {
        return new PropertySourcesPlaceholderConfigurer();
    }

}



