# Sample Application for "Introduction to ML with Apache Spark MLlib" Presentation

## Presentation
Link to the presentation: http://www.slideshare.net/tmatyashovsky/introduction-to-ml-with-apache-spark-mllib

## Idea
Create few examples to demonstrate regression, classification and clustering to Java developers.
Main focus is on feature extraction and creation of interesting ML pipelines. 

### DOU Dataset
DOU (http://dou.ua) stands for developers.org.ua is a main hub for Ukrainian developers.
It provides anonymous survey for getting information about Ukrainian engineers, their salary, experience, English level, etc. 

#### DOU Dataset Regression
Given poll results predict salary based on experience, English level and programming language.

Nuances:
* English level is string, should be converted to numeric, e.g. 0…5
* Languages are strings, should be exploded to 18 booleans, e.g. java=0|1, python=0|1, etc.
* Sparse vector [21, [0, 1, 13], [3, 4, 1]] is more preferable

#### DOU Dataset Clustering
Given poll results predict level (junior, middle, senior) based on experience and English level.

Nuances:
* English level is string, should be converted to numeric, e.g. 0…5
* 1$ difference in salary is not as significant as 1 year of experience, so data should be scaled before clustering
* Dense vector is ok  

### Pop vs. Heavy Metal
Given verse from verse1 recognize genre.

Strategy:
* Collect raw data set of verse1 (~65k sentences in total):
  * Abba, Ace of base, Backstreet Boys, Britney Spears, Christina Aguilera, Madonna, etc.
  * Black Sabbath, In Flames, Iron Maiden, Metallica, Moonspell, Nightwish, Sentenced, etc.
* Create training set, i.e. label (0|1) + features
* Train logistic regression

### MNIST Dataset
Given set of images recognize digits.

Nuances:
* Tranform images into training examples

## Build, Configure and Run

### Build
Standard build:
```
./gradlew clean build shadowJar
```
Quick build without tests:
```
./gradlew clean build shadowJar -x test
```
### Configuration
All available configuration properties are spread out via 3 files:
* application.properties - contains business logic specific stuff
* spark.properties - contains Spark specific stuff

All properties are self explanatory, but few the most important ones are listed explicitly below. 

#### Application Properties
| Name | Type | Default value | Description |
| ---- | ---- | ------------- | ----------- |
| server.port | Integer | 9090 | The port to listen for incoming HTTP requests |

#### Spark Properties
| Name | Type | Default value | Description |
| ---- | ---- | ------------- | ----------- |
| spark.master | String | spark://127.0.0.1:7077 | The URL of the Spark master. For development purposes, you can use `local[n]` that will run Spark on n threads on the local machine without connecting to a cluster. For example, `local[2]`. |
|spark.distributed-libraries | String | | Path to distributed library that should be loaded into each worker of a Spark cluster. |

#### Sample configuration for a local development environment
Create *application.properties* (for instance, in your user home directory) and override any of the described properties. 
For instance, minimum set of values that should be specified for your local environment is listed below:
```
spark.distributed-libraries=<path_to_your_repo>/spark-distributed-library/build/libs/spark-distributed-library-1.0-SNAPSHOT-all.jar

dou.training.set.csv.file.path=<path_to_your_repo>/training-set/dou/2016_may_mini.csv
dou.regression.model.directory.path=<path_to_your_repo>/training-set/dou/regression-model
dou.clustering.model.directory.path=<path_to_your_repo>/training-set/dou/clustering-model

verse1.training.set.directory.path=<path_to_your_repo>/training-set/verse1/
verse1.model.directory.path=<path_to_your_repo>/training-set/verse1/model

mnist.training.set.image.file.path=<path_to_your_repo>/training-set/mnist/train-images-idx3-ubyte
mnist.training.set.label.file.path=<path_to_your_repo>/training-set/mnist/train-labels-idx1-ubyte
mnist.test.set.image.file.path=<path_to_your_repo>/training-set/mnist/t10k-images-idx3-ubyte
mnist.test.set.label.file.path=<path_to_your_repo>/training-set/mnist/t10k-labels-idx1-ubyte

mnist.training.set.parquet.file.path=<path_to_your_repo>/training-set/mnist/training-set.parquet
mnist.test.set.parquet.file.path=<path_to_your_repo>/training-set/mnist/test-set.parquet
mnist.model.directory.path=<path_to_your_repo>/training-set/mnist/model
mnist.validation.set.directory.path=<path_to_your_repo>/training-set/mnist/validation-set

```
### Run

From your favourite IDE plese run `ApplicationConfiguration` main method. 
This will use default configuration bundled in the source code. 

In order to run the application with custom configuration please add spring.config.location parameter that corresponds to directory that contains your custom *application.properties* (in our example your user home directory). Or just enumerate them explicitly, for instance:
```
spring.config.location=/Users/<your user>/application.properties
```
