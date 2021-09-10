package ai.jgp.drsti.spark;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class DrstiLineChart extends DrstiChart {

  public DrstiLineChart(Dataset<Row> df) {
    super(df);
  }

  @Override
  public void render() {
    super.prerender();
    super.saveData();
  }

}
