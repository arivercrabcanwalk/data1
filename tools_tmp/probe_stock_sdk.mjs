import { StockSDK } from 'stock-sdk';
const sdk = new StockSDK();
for (const name of ['多氟多','埃斯顿','海南海药','大名城','立新能源','爱丽家居','一鸣食品','泛微网络']) {
  try {
    const r = await sdk.search(name);
    console.log('SEARCH', name, JSON.stringify(r).slice(0,1000));
  } catch (e) {
    console.log('ERR', name, String(e));
  }
}
