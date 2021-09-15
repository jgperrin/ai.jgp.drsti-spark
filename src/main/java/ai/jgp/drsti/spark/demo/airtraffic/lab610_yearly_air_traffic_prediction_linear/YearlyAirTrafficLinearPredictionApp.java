package ai.jgp.drsti.spark.demo.airtraffic.lab610_yearly_air_traffic_prediction_linear;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.functions.when;
import static org.apache.spark.sql.functions.year;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.RegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.jgp.drsti.spark.DrstiLineChart;
import ai.jgp.drsti.spark.DrstiUtils;
import ai.jgp.drsti.spark.utils.DataframeUtils;

/**
 * 
 * @author jgp
 *
 */
public class YearlyAirTrafficLinearPredictionApp {
  private static Logger log =
      LoggerFactory.getLogger(YearlyAirTrafficLinearPredictionApp.class);

  public static void main(String[] args) {
    YearlyAirTrafficLinearPredictionApp app =
        new YearlyAirTrafficLinearPredictionApp();
    app.start();
  }

  /**
   * Real work goes here...
   */
  private boolean start() {
    log.debug("-> start()");
    long t0 = System.currentTimeMillis();

    // Creates a session on a local master
    SparkSession spark = SparkSession.builder()
        .appName("CSV to Dataset")
        .master("local[*]")
        .getOrCreate();

    long tc = System.currentTimeMillis();
    log.info("Spark master available in {} ms.", (tc - t0));
    t0 = tc;

    // Reading gold data from Delta
    Dataset<Row> goldDf = spark.read().format("delta")
        .load("./data/tmp/airtrafficmonth_all")
        .orderBy(col("month"));

    tc = System.currentTimeMillis();
    log.info("Reading gold zone in {} ms.", (tc - t0));
    t0 = tc;

    Dataset<Row> dfYear = goldDf
        .withColumn("year", year(col("month")))
        .groupBy(col("year"))
        .agg(sum("pax").as("pax"),
            sum("internationalPax").as("internationalPax"),
            sum("domesticPax").as("domesticPax"))
        .orderBy(col("year"));

    tc = System.currentTimeMillis();
    log.info("Transformation for yearly graph {} ms.", (tc - t0));
    t0 = tc;

    dfYear.show(5, false);
    dfYear.printSchema();

    String[] inputCols = { "year" };
    VectorAssembler assembler = new VectorAssembler()
        .setInputCols(inputCols)
        .setOutputCol("features");
    dfYear = assembler.transform(dfYear);

    LinearRegression lr = new LinearRegression()
        .setMaxIter(10)
        .setRegParam(0.3)
        .setElasticNetParam(0.8).setLabelCol("pax");

    int threshold = 2019;
    Dataset<Row> trainingData =
        dfYear.filter(col("year").$less$eq(threshold));
    Dataset<Row> testData = dfYear.filter(col("year").$greater(threshold));

    LinearRegressionModel model = lr.fit(trainingData);

    // Make predictions on test data
    Dataset<Row> predictions = model.transform(testData);
    predictions.show(20);

    predict(2021, model);
    predict(2020, model);
    predict(2019, model);
    predict(2018, model);
    predict(2017, model);

    Integer[] l = new Integer[] { 2022, 2023, 2024, 2025, 2026 };
    List<Integer> data = Arrays.asList(l);
    Dataset<Row> futuresDf =
        spark.createDataset(data, Encoders.INT()).toDF()
            .withColumnRenamed("value", "year");
    assembler = new VectorAssembler()
        .setInputCols(inputCols)
        .setOutputCol("features");
    futuresDf = assembler.transform(futuresDf); 
    log.info("Futures");
    futuresDf.show();
    futuresDf.printSchema();
    log.info("/Futures");

    dfYear = dfYear.unionByName(futuresDf, true);

    dfYear = model.transform(dfYear);
    dfYear.show(30);
    dfYear.printSchema();

    // Preparing dataframe for graph
    dfYear = dfYear
        .drop("features")
        .drop("indexedFeatures")
        .drop("rawFeatures")
        .drop("internationalPax")
        .drop("domesticPax")
        .withColumn("paxInModel2019",
            when(col("year").$less$eq(threshold), col("pax"))
                .otherwise(null));

    // Graph
    dfYear = DrstiUtils.setHeader(dfYear, "year", "Year");
    dfYear = DrstiUtils.setHeader(dfYear, "paxInModel2019",
        "Passenger used in model 2019");
    dfYear = DrstiUtils.setHeader(dfYear, "pax", "Passengers");
    dfYear =
        DrstiUtils.setHeader(dfYear, "prediction", "Prediction (-> 2019)");

    DrstiLineChart d = new DrstiLineChart(dfYear);
    d.setTitle("US air traffic, in passengers, per year");
    d.setXTitle("Year " + DataframeUtils.min(dfYear, "year") + " - " +
        DataframeUtils.max(dfYear, "year"));
    d.setYTitle("Passengers (000s)");
    d.render();

    tc = System.currentTimeMillis();
    log.info("Data exported for in {} ms.", (tc - t0));

    spark.stop();
    return true;
  }

  private double predict(double feature, RegressionModel model) {
    double p = model.predict(Vectors.dense(feature));
    log.info("Passengers in year {}: {}", feature, p);
    return p;
  }
}
