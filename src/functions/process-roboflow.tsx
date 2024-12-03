export const processRoboflowResults = (roboflowData: any) => {
  const predictions = roboflowData.predictions || [];
  if (predictions.length > 0) {
    const topPrediction = predictions.reduce((prev: any, current: any) =>
      prev.confidence > current.confidence ? prev : current,
    );

    return {
      type: topPrediction.class as 'NotCoffee Leaf' | 'Coffee Leaf',
      confidence: topPrediction.confidence,
      details: `Detected: ${topPrediction.class}`,
    };
  }

  return null;
};
