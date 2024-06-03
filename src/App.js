import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import './style.css';  // Import the CSS file
import QueryComponent from './components/QueryComponent';
// Create a client
const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">

        <main>
          {/* <QueryComponent /> */}
          <QueryComponent />
        </main>
      </div>
    </QueryClientProvider>
  );
}

export default App;
