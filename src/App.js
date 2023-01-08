import logo from './logo.svg';
import './App.css';
import '../node_modules/bootstrap/dist/css/bootstrap.min.css'
import ContactForm from './components/ContactForm'

//<img src={logo} className="App-logo" alt="logo" />

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <ContactForm />
        <p>
        Edit <code>src/App.js</code> and this is the FYP.
        </p>
        <a
          className="App-link"
          href="https://www.binance.com/en"
          target="_blank"
          rel="noopener noreferrer"
        >
          click here to go to the Algo-trading platform
        </a>
      </header>
    </div>
  );
}

export default App;
