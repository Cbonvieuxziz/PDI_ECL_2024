import { useEffect, useState } from 'react'
import { LineChart, XAxis, YAxis, CartesianGrid, Line, Legend } from 'recharts'
import Papa from 'papaparse'
import csvFile from './utils/day-ahead-data.csv'
import { parseAndRound } from './utils/parseAndRound'

const NB_OF_DECIMAL_PLACES = 3

export default function Home() {

    const [chartData, setChartData] = useState();
    const [displayedChartData, setDisplayedChartData] = useState();
    
    const parseChartData = (data) => {
        const floatChartData = []

        data.forEach(row => {
            floatChartData.push({ 
                date: parseInt(row.date), 
                prediction: parseAndRound(row.prediction, NB_OF_DECIMAL_PLACES), 
                reality: parseAndRound(row.reality, NB_OF_DECIMAL_PLACES)
            })
        });
        
        setChartData(floatChartData)
        setDisplayedChartData(floatChartData)
    }

    useEffect(() => {
        Papa.parse(csvFile, {
            header: true,
            download: true,
            complete: function (input) {
                // console.log(input.data)
                parseChartData(input.data.slice(0, 60))
            }
        });
    }, [])

    return (
        <div className='mx-40 pb-10'>
            <div className="flex justify-center items-center h-32 text-4xl font-semibold">
                Electricity price predictor
            </div>

            <div className="flex justify-center">
                <Chip>
                    France
                </Chip>

                <Chip>
                    During the past two months
                </Chip>

                <Chip>
                    €
                </Chip>
            </div>

            <div className='my-8 text-xl'>
                Past predictions compared to the real data measured :
            </div>

            <div className='flex items-center justify-center my-10'>
                <LineChart width={1000} height={300} data={displayedChartData}>
                    <XAxis dataKey="date" interval={"equidistantPreserveStart"} />
                    <YAxis />
                    <CartesianGrid stroke="#eee" strokeDasharray="0" />
                    <Legend iconType='plainline' />
                    <Line type="monotone" dataKey="prediction" stroke="#002060" dot={false} strokeWidth={2} />
                    <Line type="monotone" dataKey="reality" stroke="#00FF00" dot={false} strokeWidth={2} />
                    {/*<Tooltip />*/}
                </LineChart>
            </div>

            <div className='mb-6 text-xl'>
                Future predictions :
            </div>

            <div className='text-xl'>
                Predictions for the days to come : <span className='font-semibold'>120€/MWh</span>
            </div>
        </div>
    )
}

function Chip({ children }) {
    return (
        <div className="rounded-full bg-gray-100 py-2 px-4 mx-2 min-w-20 text-center">
            {children}
        </div>
    )
}