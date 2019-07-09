/*
 * QUANTCONNECT.COM - Democratizing Finance, Empowering Individuals.
 * Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Ebby.Collections {
    /// <summary>
    /// A never ending queue that will dequeue and reenqueue the same item
    /// </summary>
    public class CircularQueue<T> : IEnumerable<T>, ICloneable {
        private readonly T _head;
        private readonly Queue<T> _queue;

        /// <summary>
        /// Fired when we do a full circle
        /// </summary>
        public event EventHandler CircleCompleted;

        /// <summary>
        /// Initializes a new instance of the <see cref="CircularQueue{T}"/> class
        /// </summary>
        /// <param name="items">The items in the queue</param>
        public CircularQueue(params T[] items)
            : this((IEnumerable<T>) items) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="CircularQueue{T}"/> class
        /// </summary>
        /// <param name="items">The items in the queue</param>
        public CircularQueue(IEnumerable<T> items) {
            _queue = new Queue<T>();

            var first = true;
            foreach (var item in items) {
                if (first) {
                    first = false;
                    _head = item;
                }

                _queue.Enqueue(item);
            }
        }

        /// <summary>
        /// Dequeues the next item
        /// </summary>
        /// <returns>The next item</returns>
        public T Dequeue() {
            var item = _queue.Dequeue();
            if (item.Equals(_head)) {
                OnCircleCompleted();
            }

            _queue.Enqueue(item);
            return item;
        }

        /// <summary>
        ///     Note: this is unsafe to use after <see cref="Dequeue"/> has been called. Might cause unexpected result.
        /// </summary>
        /// <param name="obj"></param>
        public void Enqueue(T obj) {
            _queue.Enqueue(obj);
        }

        /// <summary>
        /// Event invocator for the <see cref="CircleCompleted"/> evet
        /// </summary>
        protected virtual void OnCircleCompleted() {
            var handler = CircleCompleted;
            if (handler != null)
                handler(this, EventArgs.Empty);
        }

        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<T> GetEnumerator() {
            return _queue.GetEnumerator();
        }

        /// <summary>Returns an enumerator that iterates through a collection.</summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator" /> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator() {
            return ((IEnumerable) _queue).GetEnumerator();
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        public object Clone() {
            return new CircularQueue<T>(this);
        }
    }

    /// <summary>
    /// A never ending queue that will dequeue and reenqueue the same item
    /// </summary>
    public class CircularQueueThreadSafe<T> : IEnumerable<T>, ICloneable {
        private readonly T _head;
        private readonly Queue<T> _queue;

        /// <summary>
        /// Fired when we do a full circle
        /// </summary>
        public event EventHandler CircleCompleted;

        /// <summary>
        /// Initializes a new instance of the <see cref="CircularQueue{T}"/> class
        /// </summary>
        /// <param name="items">The items in the queue</param>
        public CircularQueueThreadSafe(params T[] items)
            : this((IEnumerable<T>) items) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="CircularQueue{T}"/> class
        /// </summary>
        /// <param name="items">The items in the queue</param>
        public CircularQueueThreadSafe(IEnumerable<T> items) {
            _queue = new Queue<T>();

            var first = true;
            foreach (var item in items) {
                if (first) {
                    first = false;
                    _head = item;
                }

                _queue.Enqueue(item);
            }
        }

        /// <summary>
        /// Dequeues the next item
        /// </summary>
        /// <returns>The next item</returns>
        public T Dequeue() {
            lock (_queue) {
                var item = _queue.Dequeue();
                if (item.Equals(_head)) {
                    OnCircleCompleted();
                }

                _queue.Enqueue(item);
                return item;
            }
        }

        /// <summary>
        ///     Note: this is unsafe to use after <see cref="Dequeue"/> has been called. Might cause unexpected result.
        /// </summary>
        /// <param name="obj"></param>
        public void Enqueue(T obj) {
            lock (_queue)
                _queue.Enqueue(obj);
        }

        /// <summary>
        /// Event invocator for the <see cref="CircleCompleted"/> evet
        /// </summary>
        protected virtual void OnCircleCompleted() {
            var handler = CircleCompleted;
            if (handler != null)
                handler(this, EventArgs.Empty);
        }

        /// <summary>Returns an enumerator that iterates through the collection.</summary>
        /// <returns>An enumerator that can be used to iterate through the collection.</returns>
        public IEnumerator<T> GetEnumerator() {
            lock (_queue)
                return _queue.ToList().GetEnumerator();
        }

        /// <summary>Returns an enumerator that iterates through a collection.</summary>
        /// <returns>An <see cref="T:System.Collections.IEnumerator" /> object that can be used to iterate through the collection.</returns>
        IEnumerator IEnumerable.GetEnumerator() {
            lock (_queue)
                return (_queue).ToArray().GetEnumerator();
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        public object Clone() {
            lock (_queue) {
                return new CircularQueue<T>(this);
            }
        }
    }
}