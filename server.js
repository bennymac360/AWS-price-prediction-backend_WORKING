// server.js

require('dotenv').config(); // Load environment variables from .env file

const express = require('express');
const cors = require('cors');
const ethers = require('ethers');
const { exec } = require('child_process');
const axios = require('axios');
const mongoose = require('mongoose');

const app = express();
const port = process.env.PORT || 5000; // Use environment variable or default to 5000

app.use(
  cors({
    origin: 'https://pricepredictor.xyz', // Your front-end domain
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    credentials: true,
  })
);

app.use(express.json());

// MongoDB connection
const mongoURI = process.env.MONGO_URI || 'mongodb://localhost:27017/priceprediction';

mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

mongoose.connection.on('connected', () => {
  console.log('MongoDB connected');
});

// Define Prediction Schema
const predictionSchema = new mongoose.Schema({
  walletAddress: {
    type: String,
    required: true,
    index: true,
  },
  coinSymbol: {
    type: String,
    required: true,
  },
  predictedPrice: {
    type: String,
    required: true,
  },
  datePredicted: {
    type: Date,
    default: Date.now,
  },
});

const Prediction = mongoose.model('Prediction', predictionSchema);

// Define networks configuration with 'enabled' flag and PORO token details
const networks = {
  '0xe': {
    chainName: 'Flare',
    symbol: 'FLR',
    decimals: 18,
    contractAddress: '0x9252daC7C660a333ac35190b58b86758c965C715', // Replace with your Flare contract address
    paymentAmount: '5',
    rpcUrl: 'https://flare-api.flare.network/ext/C/rpc',
    blockExplorerUrl: 'https://flare-explorer.flare.network',
    enabled: true,
    poroTokenAddress: '0xBB12324722a3dd74Cc1Bbc36445a170Aa1423E22', // Replace with your PORO token address on Flare
    poroTokenDecimals: 18,
  },
  '0x13': {
    chainName: 'Songbird',
    symbol: 'SGB',
    decimals: 18,
    contractAddress: 'YOUR_SONGBIRD_CONTRACT_ADDRESS', // Replace with your Songbird contract address
    paymentAmount: '10',
    rpcUrl: 'https://songbird-api.flare.network/ext/C/rpc',
    blockExplorerUrl: 'https://songbird-explorer.flare.network',
    enabled: false,
    poroTokenAddress: 'YOUR_PORO_TOKEN_ADDRESS_ON_SONGBIRD', // Replace with your PORO token address on Songbird
    poroTokenDecimals: 18,
  },
  '0x72': {
    chainName: 'Coston2',
    symbol: 'C2FLR',
    decimals: 18,
    contractAddress: '0xEcd0fAbb2825814B480156a8886fBE21D2245d97', // Replace with your Coston2 contract address
    paymentAmount: '5',
    rpcUrl: 'https://coston2-api.flare.network/ext/C/rpc',
    blockExplorerUrl: 'https://coston2-explorer.flare.network',
    enabled: false,
    poroTokenAddress: '0x469dd82993185fb3274c1A53E517ff29454f8a6e', // Replace with your PORO token address on Coston2
    poroTokenDecimals: 18,
  },
  '0x1': {
    chainName: 'Ethereum',
    symbol: 'ETH',
    decimals: 18,
    contractAddress: 'YOUR_ETHEREUM_CONTRACT_ADDRESS', // Replace with your Ethereum contract address if applicable
    paymentAmount: '0.0004',
    rpcUrl: 'https://ethereum-rpc.publicnode.com',
    blockExplorerUrl: 'https://etherscan.io',
    enabled: false,
    poroTokenAddress: '',
    poroTokenDecimals: 18,
  },
  '0xa86a': {
    chainName: 'Avalanche',
    symbol: 'AVAX',
    decimals: 18,
    contractAddress: 'YOUR_AVALANCHE_CONTRACT_ADDRESS', // Replace with your Avalanche contract address if applicable
    paymentAmount: '0.005',
    rpcUrl: 'https://api.avax.network/ext/bc/C/rpc',
    blockExplorerUrl: 'https://snowtrace.io',
    enabled: false,
    poroTokenAddress: '',
    poroTokenDecimals: 18,
  },
};

// Contract ABI for the updated PricePrediction contract
const contractABI = [
  {
    inputs: [],
    name: 'makePaymentInFLR',
    outputs: [],
    stateMutability: 'payable',
    type: 'function',
  },
  {
    inputs: [],
    name: 'makePaymentInPORO',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [],
    stateMutability: 'nonpayable',
    type: 'constructor',
  },
  {
    anonymous: false,
    inputs: [
      {
        indexed: true,
        internalType: 'address',
        name: 'user',
        type: 'address',
      },
      {
        indexed: false,
        internalType: 'uint256',
        name: 'amount',
        type: 'uint256',
      },
      {
        indexed: false,
        internalType: 'string',
        name: 'paymentMethod',
        type: 'string',
      },
    ],
    name: 'PaymentReceived',
    type: 'event',
  },
  {
    inputs: [],
    name: 'withdraw',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [],
    name: 'owner',
    outputs: [
      {
        internalType: 'address payable',
        name: '',
        type: 'address',
      },
    ],
    stateMutability: 'view',
    type: 'function',
  },
];

// Endpoint to receive coinId, model, transactionHash, walletAddress, chainId, and paymentMethod
app.post('/api/request-prediction', async (req, res) => {
  const {
    coinId,
    model,
    transactionHash,
    walletAddress,
    chainId,
    paymentMethod,
  } = req.body;

  if (
    coinId === undefined ||
    model === undefined ||
    transactionHash === undefined ||
    walletAddress === undefined ||
    chainId === undefined ||
    paymentMethod === undefined
  ) {
    return res.status(400).json({ error: 'Missing required parameters' });
  }

  // Validate wallet address format
  if (!ethers.utils.isAddress(walletAddress)) {
    return res.status(400).json({ error: 'Invalid wallet address' });
  }

  // Validate model
  if (![0, 1].includes(model)) {
    return res.status(400).json({ error: 'Invalid model selection' });
  }

  // Validate paymentMethod
  if (!['FLR', 'PORO'].includes(paymentMethod)) {
    return res.status(400).json({ error: 'Invalid payment method' });
  }

  // Validate chainId and get network configuration
  const network = networks[chainId.toLowerCase()];
  if (!network || !network.enabled) {
    return res.status(400).json({ error: 'Unsupported or disabled network' });
  }

  try {
    // Create provider for the selected network
    const provider = new ethers.providers.JsonRpcProvider(network.rpcUrl);

    // Create contract instance for the selected network
    const contract = new ethers.Contract(network.contractAddress, contractABI, provider);

    // Verify the transaction
    const isValid = await verifyTransaction(
      transactionHash,
      walletAddress,
      provider,
      network.contractAddress,
      network.paymentAmount,
      network.decimals,
      paymentMethod,
      network.poroTokenAddress,
      network.poroTokenDecimals
    );

    if (!isValid) {
      return res.status(400).json({ error: 'Invalid transaction' });
    }

    // Run the Python script with coinId and model
    const predictedPrice = await runPythonScript(coinId, model);

    // Save the prediction to the database
    await savePrediction(walletAddress, coinId, predictedPrice);

    res.json({ predictedPrice });
  } catch (error) {
    console.error('Error processing prediction:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Verify transaction
async function verifyTransaction(
  transactionHash,
  walletAddress,
  provider,
  contractAddress,
  expectedAmount,
  decimals,
  paymentMethod,
  poroTokenAddress,
  poroTokenDecimals
) {
  try {
    const tx = await provider.getTransaction(transactionHash);

    if (!tx) {
      console.error('Transaction not found');
      return false;
    }

    if (tx.from.toLowerCase() !== walletAddress.toLowerCase()) {
      console.error(`Transaction was sent from ${tx.from}, expected ${walletAddress}`);
      return false;
    }

    const receipt = await provider.getTransactionReceipt(transactionHash);
    if (!receipt || receipt.confirmations < 1) {
      console.error('Transaction not yet confirmed');
      return false;
    }

    if (paymentMethod === 'FLR') {
      if (tx.to.toLowerCase() !== contractAddress.toLowerCase()) {
        console.error(`Transaction was sent to ${tx.to}, expected ${contractAddress}`);
        return false;
      }

      // Parse expected amount into a BigNumber
      const expectedAmountBN = ethers.utils.parseUnits(expectedAmount.toString(), decimals);

      // Compare BigNumbers
      if (!tx.value.eq(expectedAmountBN)) {
        console.error(
          `Incorrect transaction amount: expected ${expectedAmountBN.toString()}, got ${tx.value.toString()}`
        );
        return false;
      }

      // Check if the method called is makePaymentInFLR
      const iface = new ethers.utils.Interface(contractABI);
      const decodedData = iface.parseTransaction({ data: tx.data });

      if (decodedData.name !== 'makePaymentInFLR') {
        console.error('Transaction is not a call to makePaymentInFLR');
        return false;
      }
    } else if (paymentMethod === 'PORO') {
      // Check if the transaction is a contract interaction with makePaymentInPORO
      if (tx.to.toLowerCase() !== contractAddress.toLowerCase()) {
        console.error(`Transaction was sent to ${tx.to}, expected ${contractAddress}`);
        return false;
      }

      // Parse the transaction input data
      const iface = new ethers.utils.Interface(contractABI);
      const decodedData = iface.parseTransaction({ data: tx.data });

      if (decodedData.name !== 'makePaymentInPORO') {
        console.error('Transaction is not a call to makePaymentInPORO');
        return false;
      }

      // Optionally, verify that the PORO tokens were transferred to the contract
      // This can be complex and may require additional logic, such as checking the events
      // emitted by the PORO token contract. For simplicity, we're assuming the transaction
      // is valid if it successfully calls makePaymentInPORO.
    }

    console.log('Transaction verified successfully');
    return true;
  } catch (error) {
    console.error('Error verifying transaction:', error);
    return false;
  }
}

// Function to save prediction
async function savePrediction(walletAddress, coinId, predictedPrice) {
  try {
    // Get the coin symbol
    const coinSymbol = await getCoinSymbol(coinId);

    // Create a new prediction document
    const newPrediction = new Prediction({
      walletAddress: walletAddress.toLowerCase(),
      coinSymbol,
      predictedPrice,
    });

    // Save the prediction
    await newPrediction.save();

    // Limit to last 5 predictions
    const predictions = await Prediction.find({ walletAddress: walletAddress.toLowerCase() })
      .sort({ datePredicted: -1 })
      .skip(5);

    // Remove older predictions
    for (const prediction of predictions) {
      await Prediction.deleteOne({ _id: prediction._id });
    }
  } catch (error) {
    console.error('Error saving prediction:', error);
    throw error;
  }
}

// Function to get coin symbol from coinId
async function getCoinSymbol(coinId) {
  // For simplicity, assume coinId is the symbol in uppercase
  return coinId.toUpperCase();
}

// Endpoint to fetch top 200 coins by market cap
app.get('/api/top-coins', async (req, res) => {
  try {
    const coinIds = await getTopCoinsByMarketCap(200);
    res.json({ coinIds });
  } catch (error) {
    console.error('Error fetching top coins:', error);
    res.status(500).json({ error: 'Failed to fetch top coins' });
  }
});

// Function to fetch top coins
async function getTopCoinsByMarketCap(n = 200) {
  const url = 'https://api.coingecko.com/api/v3/coins/markets';
  const params = {
    vs_currency: 'usd',
    order: 'market_cap_desc',
    per_page: n,
    page: 1,
    sparkline: 'false',
  };
  const headers = {
    accept: 'application/json',
    'x-cg-demo-api-key': 'CG-ZZLFUQooopRkr47Z1yqNWKyP',
  };
  try {
    const response = await axios.get(url, { params, headers });
    if (response.status === 200) {
      const coins = response.data;
      const coin_ids = coins.map((coin) => ({
        id: coin.id,
        name: coin.name,
        symbol: coin.symbol,
        image: coin.image,
      }));
      return coin_ids;
    } else {
      console.error(`Error fetching top coins: ${response.status}, ${response.statusText}`);
      throw new Error('Failed to fetch top coins');
    }
  } catch (error) {
    console.error(`Error in getTopCoinsByMarketCap: ${error}`);
    throw error;
  }
}

// Run Python script with coinId and model
function runPythonScript(coinId, model) {
  return new Promise((resolve, reject) => {
    const command = `python "E:\\Amazon_Price_Predict.py" ${coinId} ${model}`;
    console.log('Executing command:', command);

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error('Execution error:', error);
        return reject(`Execution error: ${error.message}`);
      }
      if (stderr) {
        console.error('Script error:', stderr);
        // Note: Some scripts output warnings to stderr even when successful
      }

      // Assume the script prints the predicted price
      const predictedPrice = stdout.trim();
      resolve(predictedPrice);
    });
  });
}

// Endpoint to retrieve predictions for a wallet address
app.get('/api/predictions/:walletAddress', async (req, res) => {
  const walletAddress = req.params.walletAddress.toLowerCase();

  // Validate wallet address format
  if (!ethers.utils.isAddress(walletAddress)) {
    return res.status(400).json({ error: 'Invalid wallet address' });
  }

  try {
    const predictions = await Prediction.find({ walletAddress })
      .sort({ datePredicted: -1 })
      .limit(5);

    res.json({ predictions });
  } catch (error) {
    console.error('Error fetching predictions:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(port, () => {
  console.log(`Backend server is running on port ${port}`);
});
